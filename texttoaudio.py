#This file take the text and transform in a audio file with the announcer (.npz file)
from scipy.io.wavfile import write as write_wav

from bark.api import generate_audio
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic

def generate_with_settings(text_prompt, semantic_temp=0.7, semantic_top_k=50, semantic_top_p=0.95, coarse_temp=0.7, coarse_top_k=50, coarse_top_p=0.95, fine_temp=0.5, voice_name=None, use_semantic_history_prompt=True, use_coarse_history_prompt=True, use_fine_history_prompt=True, output_full=False):
    # generation with more control
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=voice_name if use_semantic_history_prompt else None,
        temp=semantic_temp,
        top_k=semantic_top_k,
        top_p=semantic_top_p,
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=voice_name if use_coarse_history_prompt else None,
        temp=coarse_temp,
        top_k=coarse_top_k,
        top_p=coarse_top_p,
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=voice_name if use_fine_history_prompt else None,
        temp=fine_temp,
    )

    if output_full:
        full_generation = {
            'semantic_prompt': x_semantic,
            'coarse_prompt': x_coarse_gen,
            'fine_prompt': x_fine_gen,
        }
        return full_generation, codec_decode(x_fine_gen)
    return codec_decode(x_fine_gen)


text_prompt = output("Digite o texto que deve ser transformado em voz:") #The user write what must to be transformed to audio.
filepath = output("Escolha o nome do arquivo:") # change this to your desired output path
announcer = output("Escolha o nome do locutor (arquivo .npz sem extensão:)")

audio_array = generate_with_settings( #This is configuration
    text_prompt,
    semantic_temp=0.7,
    semantic_top_k=50,
    semantic_top_p=0.99,
    coarse_temp=0.7,
    coarse_top_k=50,
    coarse_top_p=0.95,
    fine_temp=0.5,
    voice_name= announcer + ".npz",
    use_semantic_history_prompt=False,
    use_coarse_history_prompt=True,
    use_fine_history_prompt=True,
    output_full=False
)

write_wav(filepath, SAMPLE_RATE, audio_array)
