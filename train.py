import numpy as np
import torch
import torchaudio
from bark.generation import SAMPLE_RATE, load_codec_model

from customtokenizer import CustomTokenizer
from hubert_manager import HuBERTManager
from pre_kmeans_hubert import CustomHubert
from encodec.utils import convert_audio
from typing import Union

import bark.generation as o
from bark.generation import *

#configuration
bark_offload_cpu = True
bark_use_cpu = False
bark_half = False
bark_models_mix={
    'text': {'name': 'large', 'large': True},
    'coarse':{'name': 'large', 'large': True},
    'fine':{'name': 'large', 'large': True}
    }

############# webui\modules\implementations\patches\bark_generation.py ####################
SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

ALLOWED_PROMPTS = ["announcer"]
for _, lang in SUPPORTED_LANGS:
    for prefix in ("", f"v2{os.path.sep}"):
        for n in range(10):
            ALLOWED_PROMPTS.append(f"{prefix}{lang}_speaker_{n}")
for n in range(10):
    ALLOWED_PROMPTS.append(f"speaker_{n}")


def generate_text_semantic_new(
        text,
        history_prompt: Union[str, dict] = None,
        temp=0.7,
        top_k=None,
        top_p=None,
        silent=False,
        min_eos_p=0.2,
        max_gen_duration_s=None,
        allow_early_stop=True,
        use_kv_caching=False,
):
    """Generate semantic tokens from text."""
    assert isinstance(text, str)
    text = o._normalize_whitespace(text)
    # assert len(text.strip()) > 0
    if history_prompt is not None:
        skip = False
        if isinstance(history_prompt, dict):
            semantic_history = history_prompt['semantic_prompt']
        elif history_prompt.endswith(".npz"):
            semantic_history = np.load(history_prompt)["semantic_prompt"]
        else:
            if history_prompt in ALLOWED_PROMPTS:
                semantic_history = np.load(
                    os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt}.npz")
                )["semantic_prompt"]
            else:
                filename = f'data/bark_custom_speakers/{history_prompt}.npz'
                if os.path.isfile(filename):
                    semantic_history = np.load(
                        filename
                    )["semantic_prompt"]
                else:
                    skip = True
        if not skip:
            assert (
                    isinstance(semantic_history, np.ndarray)
                    and len(semantic_history.shape) == 1
                    and len(semantic_history) > 0
                    and semantic_history.min() >= 0
                    and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            )
        else:
            semantic_history = None
    else:
        semantic_history = None
    # load models if not yet exist
    global models
    global models_devices
    if "text" not in models:
        preload_models_new()
    model_container = models["text"]
    model = model_container["model"]
    tokenizer = model_container["tokenizer"]
    encoded_text = np.array(o._tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if bark_offload_cpu:
        model.to(models_devices["text"])
    device = next(model.parameters()).device
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = torch.from_numpy(
        np.hstack([
            encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
        ]).astype(np.int64)
    )[None]
    assert x.shape[1] == 256 + 256 + 1
    with o._inference_mode():
        x = x.to(device)
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        # pbar = tqdm.tqdm(disable=silent, total=100, desc='Generating semantics...')
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )
            if top_p is not None:
                # faster to convert to numpy
                logits_device = relevant_logits.device
                logits_dtype = relevant_logits.type()
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = F.softmax(relevant_logits / temp, dim=-1)
            # multinomial bugged on mps: shuttle to cpu if necessary
            inf_device = probs.device
            if probs.device.type == "mps":
                probs = probs.to("cpu")
            item_next = torch.multinomial(probs, num_samples=1)
            probs = probs.to(inf_device)
            item_next = item_next.to(inf_device)
            if allow_early_stop and (
                    item_next == SEMANTIC_VOCAB_SIZE
                    or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                # pbar.update(100 - pbar_state)
                progress((pbar_state, 100), desc='Generating semantics...')
                break
            x = torch.cat((x, item_next[None]), dim=1)
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                # pbar.update(100 - pbar_state)
                progress((pbar_state, 100), desc='Generating semantics...')
                break
            if n == n_tot_steps - 1:
                # pbar.update(100 - pbar_state)
                progress((pbar_state, 100), desc='Generating semantics...')
                break
            del logits, relevant_logits, probs, item_next
            req_pbar_state = np.min([100, int(round(100 * n / n_tot_steps))])
            if req_pbar_state > pbar_state:
                # pbar.update(req_pbar_state - pbar_state)
                progress((pbar_state, req_pbar_state), desc='Generating semantics...')
            pbar_state = req_pbar_state
        # pbar.close()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1:]
    if bark_offload_cpu:
        model.to("cpu")
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    o._clear_cuda_cache()
    return out


def generate_coarse_new(
        x_semantic,
        history_prompt: Union[str, dict] = None,
        temp=0.7,
        top_k=None,
        top_p=None,
        silent=False,
        max_coarse_history=630,  # min 60 (faster), max 630 (more context)
        sliding_window_len=60,
        use_kv_caching=False,
):
    """Generate coarse audio codes from semantic tokens."""
    assert (
            isinstance(x_semantic, np.ndarray)
            and len(x_semantic.shape) == 1
            and len(x_semantic) > 0
            and x_semantic.min() >= 0
            and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if history_prompt is not None:
        skip = False
        if isinstance(history_prompt, dict):
            x_history = history_prompt
        elif history_prompt.endswith(".npz"):
            x_history = np.load(history_prompt)
        else:
            if history_prompt in ALLOWED_PROMPTS:
                x_history = np.load(
                    os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt}.npz")
                )
            else:
                filename = f'data/bark_custom_speakers/{history_prompt}.npz'
                if os.path.isfile(filename):
                    x_history = np.load(
                        filename
                    )
                else:
                    skip = True
        if not skip:
            x_semantic_history = x_history["semantic_prompt"]
            x_coarse_history = x_history["coarse_prompt"]
            assert (
                    isinstance(x_semantic_history, np.ndarray)
                    and len(x_semantic_history.shape) == 1
                    and len(x_semantic_history) > 0
                    and x_semantic_history.min() >= 0
                    and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
                    and isinstance(x_coarse_history, np.ndarray)
                    and len(x_coarse_history.shape) == 2
                    and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
                    and x_coarse_history.shape[-1] >= 0
                    and x_coarse_history.min() >= 0
                    and x_coarse_history.max() <= CODEBOOK_SIZE - 1
                    # and (
                    #         round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                    #         == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
                    # )
            )
        x_coarse_history = o._flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)
    # load models if not yet exist
    global models
    global models_devices
    if "coarse" not in models:
        preload_models_new()
    model = models["coarse"]
    if bark_offload_cpu:
        model.to(models_devices["coarse"])
    device = next(model.parameters()).device
    # start loop
    n_steps = int(
        round(
            np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
            * N_COARSE_CODEBOOKS
        )
    )
    assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    with o._inference_mode():
        x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
        x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = 0
        for curr_step in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent, desc='Generating coarse audio...'):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]):]
            x_in = x_in[:, :256]
            x_in = F.pad(
                x_in,
                (0, 256 - x_in.shape[-1]),
                "constant",
                COARSE_SEMANTIC_PAD_TOKEN,
            )
            x_in = torch.hstack(
                [
                    x_in,
                    torch.tensor([COARSE_INFER_TOKEN])[None].to(device),
                    x_coarse_in[:, -max_coarse_history:],
                ]
            )
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [-1]]
                else:
                    x_input = x_in

                logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                logit_start_idx = (
                        SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = (
                        SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                )
                relevant_logits = logits[0, 0, logit_start_idx:logit_end_idx]
                if top_p is not None:
                    # faster to convert to numpy
                    logits_device = relevant_logits.device
                    logits_dtype = relevant_logits.type()
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temp, dim=-1)
                # multinomial bugged on mps: shuttle to cpu if necessary
                inf_device = probs.device
                if probs.device.type == "mps":
                    probs = probs.to("cpu")
                item_next = torch.multinomial(probs, num_samples=1)
                probs = probs.to(inf_device)
                item_next = item_next.to(inf_device)
                item_next += logit_start_idx
                x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                x_in = torch.cat((x_in, item_next[None]), dim=1)
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
        del x_semantic_in
    if bark_offload_cpu:
        model.to("cpu")
    gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history):]
    del x_coarse_in
    assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
    o._clear_cuda_cache()
    return gen_coarse_audio_arr


def generate_fine_new(
        x_coarse_gen,
        history_prompt: Union[str, dict] = None,
        temp=0.5,
        silent=False,
):
    """Generate full audio codes from coarse audio codes."""
    assert (
            isinstance(x_coarse_gen, np.ndarray)
            and len(x_coarse_gen.shape) == 2
            and 1 <= x_coarse_gen.shape[0] <= N_FINE_CODEBOOKS - 1
            and x_coarse_gen.shape[1] > 0
            and x_coarse_gen.min() >= 0
            and x_coarse_gen.max() <= CODEBOOK_SIZE - 1
    )
    if history_prompt is not None:
        skip = False
        if isinstance(history_prompt, dict):
            x_fine_history = history_prompt['fine_prompt']
        elif history_prompt.endswith(".npz"):
            x_fine_history = np.load(history_prompt)["fine_prompt"]
        else:
            if history_prompt in ALLOWED_PROMPTS:
                x_fine_history = np.load(
                    os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt}.npz")
                )["fine_prompt"]
            else:
                filename = f'data/bark_custom_speakers/{history_prompt}.npz'
                if os.path.isfile(filename):
                    x_fine_history = np.load(
                        filename
                    )["fine_prompt"]
                else:
                    skip = True
        if not skip:
            assert (
                    isinstance(x_fine_history, np.ndarray)
                    and len(x_fine_history.shape) == 2
                    and x_fine_history.shape[0] == N_FINE_CODEBOOKS
                    and x_fine_history.shape[1] >= 0
                    and x_fine_history.min() >= 0
                    and x_fine_history.max() <= CODEBOOK_SIZE - 1
            )
    else:
        x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    # load models if not yet exist
    global models
    global models_devices
    if "fine" not in models:
        preload_models_new()
    model = models["fine"]
    if bark_offload_cpu:
        model.to(models_devices["fine"])
    device = next(model.parameters()).device
    # make input arr
    in_arr = np.vstack(
        [
            x_coarse_gen,
            np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + CODEBOOK_SIZE,  # padding
        ]
    ).astype(np.int32)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        x_fine_history = x_fine_history.astype(np.int32)
        in_arr = np.hstack(
            [
                x_fine_history[:, -512:].astype(np.int32),
                in_arr,
            ]
        )
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = np.hstack(
            [
                in_arr,
                np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
            ]
        )
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1
    with o._inference_mode():
        in_arr = torch.tensor(in_arr.T).to(device)
        for n in tqdm.tqdm(range(n_loops), disable=silent, desc='Generating fine audio...'):
            start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
            start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx: start_idx + 1024, :][None]
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = model(nn, in_buffer)
                if temp is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                    probs = F.softmax(relevant_logits, dim=-1)
                    # multinomial bugged on mps: shuttle to cpu if necessary
                    inf_device = probs.device
                    if probs.device.type == "mps":
                        probs = probs.to("cpu")
                    codebook_preds = torch.hstack(
                        [
                            torch.multinomial(probs[nnn], num_samples=1).to(inf_device)
                            for nnn in range(rel_start_fill_idx, 1024)
                        ]
                    )
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                del logits, codebook_preds
            # transfer over info into model_in and convert to numpy
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                in_arr[
                start_fill_idx: start_fill_idx + (1024 - rel_start_fill_idx), nn
                ] = in_buffer[0, rel_start_fill_idx:, nn]
            del in_buffer
        gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
        del in_arr
    if bark_offload_cpu:
        model.to("cpu")
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    o._clear_cuda_cache()
    return gen_fine_arr


def codec_decode_new(fine_tokens, decode_on_cpu=False):
    """Turn quantized audio codes into audio array using encodec."""
    # load models if not yet exist
    global models
    global models_devices
    if "codec" not in models:
        preload_models_new()
    model = models["codec"]
    if bark_offload_cpu and not decode_on_cpu:
        model.to(models_devices["codec"])
    elif decode_on_cpu:
        model.to('cpu')
    device = next(model.parameters()).device
    arr = torch.from_numpy(fine_tokens)[None]
    arr = arr.to(device)
    arr = arr.transpose(0, 1)
    emb = model.quantizer.decode(arr)
    out = model.decoder(emb)
    audio_arr = out.detach().cpu().numpy().squeeze()
    del arr, emb, out
    if bark_offload_cpu and not decode_on_cpu:
        model.to("cpu")
    elif decode_on_cpu:
        model.to('cpu' if bark_use_cpu else 'cuda')
    return audio_arr


def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    if model_type == "text":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "coarse":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "fine":
        ConfigClass = FineGPTConfig
        ModelClass = FineGPT
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    model_info = REMOTE_MODEL_PATHS[model_key]
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        o._download(model_info["repo_id"], model_info["file_name"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    gptconf = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(gptconf)
    if bark_half:
        model = model.half()
    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    logger.info(f"model loaded: {round(n_params/1e6,1)}M params, {round(val_loss,3)} loss")
    model.eval()
    model.to(device)
    del checkpoint, state_dict
    o._clear_cuda_cache()
    if model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        return {
            "model": model,
            "tokenizer": tokenizer,
        }
    return model


def load_model(use_gpu=True, use_small=False, force_reload=False, model_type="text"):
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()

    if bark_models_mix:
        use_small = not bark_models_mix[model_type]['large']

    _load_model_f = funcy.partial(_load_model, model_type=model_type, use_small=use_small)

    global models
    global models_devices
    device = o._grab_best_device(use_gpu=use_gpu)
    model_key = f"{model_type}"
    if bark_offload_cpu:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        ckpt_path = o._get_ckpt_path(model_type, use_small=use_small)
        clean_models(model_key=model_key)
        model = _load_model_f(ckpt_path, device)
        models[model_key] = model
    if model_type == "text":
        models[model_key]["model"].to(device)
    else:
        models[model_key].to(device)
    return models[model_key]


def load_codec_model(use_gpu=True, force_reload=False):
    global models
    global models_devices
    device = o._grab_best_device(use_gpu=use_gpu)
    if device == "mps":
        # encodec doesn't support mps
        device = "cpu"
    model_key = "codec"
    #if bark_offload_cpu:
    #    models_devices[model_key] = device
    #    device = "cpu"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)
        model = o._load_codec_model(device)
        models[model_key] = model
    models[model_key].to(device)
    return models[model_key]


def preload_models_new(
    text_use_gpu=True,
    text_use_small=False,
    coarse_use_gpu=True,
    coarse_use_small=False,
    fine_use_gpu=True,
    fine_use_small=False,
    codec_use_gpu=True,
    force_reload=False,
):
    """Load all the necessary models for the pipeline."""
    if o._grab_best_device() == "cpu" and (
        text_use_gpu or coarse_use_gpu or fine_use_gpu or codec_use_gpu
    ):
        logger.warning("No GPU being used. Careful, inference might be very slow!")
    _ = load_model(
        model_type="text", use_gpu=text_use_gpu, use_small=text_use_small, force_reload=force_reload
    )
    progress((1, 4), desc='Loaded text...')
    _ = load_model(
        model_type="coarse",
        use_gpu=coarse_use_gpu,
        use_small=coarse_use_small,
        force_reload=force_reload,
    )
    progress((2, 4), desc='Loaded coarse...')
    _ = load_model(
        model_type="fine", use_gpu=fine_use_gpu, use_small=fine_use_small, force_reload=force_reload
    )
    progress((3, 4), desc='Loaded fine...')
    _ = load_codec_model(use_gpu=codec_use_gpu, force_reload=force_reload)
    progress((4, 4), desc='Loaded encodec...')


#######                 Arquivo \webui\modules\implementations\patches\bark_custom_voices.py                      ###################
def generate_semantic_fine(transcript='There actually isn\'t a way to do that. It\'s impossible. Please don\'t even bother.'):
    """
    Creates a speech file with semantics and fine audio
    :param transcript: The transcript.
    :return: tuple with (semantic, fine)
    """
    semantic = generate_text_semantic_new(transcript)  # We need speech patterns
    coarse = generate_coarse_new(semantic)  # Voice doesn't matter
    fine = generate_fine_new(coarse)  # Good audio, ready for what comes next
    return semantic, fine


huberts = {}

def load_hubert(clone_model):
    global huberts
    hubert_path = HuBERTManager.make_sure_hubert_installed()
    # model = ('quantifier_V1_hubert_base_ls960_23.pth', 'tokenizer_large.pth') if args.bark_cloning_large_model else ('quantifier_hubert_base_ls960_14.pth', 'tokenizer.pth')
    tokenizer_path = HuBERTManager.make_sure_tokenizer_installed(model=clone_model['file'], local_file=clone_model['dlfilename'], repo=clone_model['repo'])
    if 'hubert' not in huberts:
        print('Loading HuBERT')
        huberts['hubert'] = CustomHubert(hubert_path)
    if 'tokenizer' not in huberts or ('tokenizer_name' in huberts and huberts['tokenizer_name'] != clone_model['name'].casefold()):
        print('Loading Custom Tokenizer')
        tokenizer = CustomTokenizer.load_from_checkpoint(tokenizer_path, map_location=torch.device('cpu'))
        huberts['tokenizer'] = tokenizer
        huberts['tokenizer_name'] = clone_model['name'].casefold()


def wav_to_semantics(file, clone_model) -> torch.Tensor:
    # Vocab size is 10,000.

    load_hubert(clone_model)

    wav, sr = torchaudio.load(file)
    # sr, wav = wavfile.read(file)
    # wav = torch.tensor(wav, dtype=torch.float32)

    if wav.shape[0] == 2:  # Stereo to mono if needed
        wav = wav.mean(0, keepdim=True)
    if wav.shape[1] == 2:
        wav = wav.mean(1, keepdim=False).unsqueeze(-1)

    # Extract semantics in HuBERT style
    print('Extracting semantics')
    semantics = huberts['hubert'].forward(wav, input_sample_hz=sr)
    print('Tokenizing semantics')
    tokens = huberts['tokenizer'].get_token(semantics)
    return tokens


def eval_semantics(code):
    """
    BE CAREFUL, this will execute :code:
    :param code: The code to evaluate, out local will be used for the output.
    :return: The created numpy array.
    """
    _locals = locals()
    exec(code, globals(), _locals)
    return _locals['out']


def generate_course_history(fine_history):
    return fine_history[:2, :]


def generate_fine_from_wav(file):
    model = load_codec_model(use_gpu=not bark_use_cpu)  # Don't worry about reimporting, it stores the loaded model in a dict
    wav, sr = torchaudio.load(file)
    wav = convert_audio(wav, sr, SAMPLE_RATE, model.channels)
    wav = wav.unsqueeze(0)
    if not bark_use_cpu:
        wav = wav.to('cuda')
    with torch.no_grad():
        encoded_frames = model.encode(wav)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()

    codes = codes.cpu().numpy()

    return codes

###################### Arquivo webui\modules\implementations\ttsmodels.py ############################################
hubert_models_cache = None
def get_cloning_models():
        global hubert_models_cache
        if hubert_models_cache:
            return hubert_models_cache
        try:
            r = requests.get('https://raw.githubusercontent.com/gitmylo/Voice-cloning-quantizers/main/models.json')
            hubert_models_cache = r.json()
        except:  # No internet connection or something similar
            hubert_models_cache = [
            {
                "name": "Base English",
                "repo": "GitMylo/bark-voice-cloning",
                "file": "quantifier_hubert_base_ls960_14.pth",
                "language": "ENG",
                "author": "https://github.com/gitmylo/",
                "quant_version": 0,
                "official": True,
                "dlfilename": "tokenizer.pth",
                "extra": {
                  "dataset": "https://huggingface.co/datasets/GitMylo/bark-semantic-training"
                }
            },
            {
                "name": "Large English",
                "repo": "GitMylo/bark-voice-cloning",
                "file": "quantifier_V1_hubert_base_ls960_23.pth",
                "language": "ENG",
                "author": "https://github.com/gitmylo/",
                "quant_version": 1,
                "official": True,
                "dlfilename": "tokenizer_large.pth",
                "extra": {
                  "dataset": "https://huggingface.co/datasets/GitMylo/bark-semantic-training"
                }
            },
            {
                "name": "Polish",
                "repo": "Hobis/bark-voice-cloning-polish-HuBERT-quantizer",
                "file": "polish-HuBERT-quantizer_8_epoch.pth",
                "language": "POL",
                "author": "https://github.com/HobisPL",
                "quant_version": 1,
                "official": False,
                "dlfilename": "tokenizer_pol.pth",
                "extra": {
                  "dataset": "https://huggingface.co/datasets/Hobis/bark-polish-semantic-wav-training"
                }
            },
            {
                "name": "German",
                "repo": "CountFloyd/bark-voice-cloning-german-HuBERT-quantizer",
                "file": "german-HuBERT-quantizer_14_epoch.pth",
                "language": "GER",
                "author": "https://github.com/C0untFloyd",
                "quant_version": 1,
                "official": False,
                "dlfilename": "tokenizer_ger.pth",
                "extra": {
                  "dataset": "https://huggingface.co/datasets/CountFloyd/bark-german-semantic-wav-training"
                }
            },
            {
                "name": "Spanish",
                "repo": "Lancer1408/bark-es-tokenizer",
                "file": "es_tokenizer.pth",
                "language": "SPA",
                "author": "https://github.com/MaxLanc",
                "quant_version": 1,
                "official": False,
                "dlfilename": "tokenizer_spa.pth",
                "extra": {
                  "dataset": "https://huggingface.co/datasets/Lancer1408/bark-spa-token-trainy"
                }
            },
            {
                "name": "Portuguese",
                "repo": "MadVoyager/bark-voice-cloning-portuguese-HuBERT-quantizer",
                "file": "portuguese-HuBERT-quantizer_24_epoch.pth",
                "language": "POR",
                "author": "https://github.com/Subarasheese",
                "quant_version": 1,
                "official": False,
                "dlfilename": "tokenizer_por.pth",
                "extra": {
                  "dataset": "https://huggingface.co/datasets/MadVoyager/bark-portuguese-semantic-wav-training/"
                }
            },
            {
                "name": "Japanese",
                "repo": "junwchina/bark-voice-cloning-japanese-HuBERT-quantizer",
                "file": "japanese-HuBERT-quantizer_24_epoch.pth",
                "language": "JA",
                "author": "https://github.com/junwchina",
                "quant_version": 1,
                "official": False,
                "dlfilename": "tokenizer_ja.pth",
                "extra": {
                  "dataset": "https://huggingface.co/datasets/junwchina/bark-japanese-semantic-wav-training"
                }
            },
            {
                "name": "Turkish",
                "repo": "egeadam/bark-voice-cloning-turkish-HuBERT-quantizer",
                "file": "turkish_model_epoch_14.pth",
                "language": "TUR",
                "author": "https://github.com/ege-adam",
                "quant_version": 1,
                "official": False,
                "dlfilename": "tokenizer_tur.pth"
            }
            ]
        return hubert_models_cache


def create_voice(file, clone_model): #Clone model is Portuguese
	clone_model_obj = [model for model in hubert_models_cache if model['name'].casefold() == clone_model.casefold()][0]
	file_name = 'teste'
	out_file = f'{file_name}.npz'

	semantic_prompt = wav_to_semantics(file, clone_model_obj)
	fine_prompt = generate_fine_from_wav(file)
	coarse_prompt = generate_course_history(fine_prompt)


	np.savez(out_file,
			 semantic_prompt=semantic_prompt,
			 fine_prompt=fine_prompt,
			 coarse_prompt=coarse_prompt
			 )
	return file_name
    
get_cloning_models()    
arquivo = create_voice("Diogooriginal.mp3", "Portuguese")
	
