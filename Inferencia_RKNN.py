# -*- coding: utf-8 -*-
import numpy as np
import argparse
import soundfile as sf
import onnxruntime # Necesario para ONNX si se usa como fallback o comparación
import torch
import torch.nn.functional as F
import scipy.signal # Usar scipy.signal explícitamente
import os
# import base64 # Asegúrate de importar si usas base64_decode
from rknnlite.api import RKNNLite # Importar para inferencia en dispositivo

# --- Constantes del Modelo Whisper ---
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
MAX_LENGTH = 3000 # Frames Mel para 30s
N_MELS = 80
# ### CAMBIO RKNNLite ###: Añadir potencial escalado (ajustar si es necesario)
# El ejemplo usaba 0.5 para FP16. Podría no ser necesario o necesitar otro valor.
FIXED_DECODER_INPUT_LENGTH = 12
INPUT_SCALE = 1 # O el valor usado en la conversión

# --- Funciones de Preprocesamiento de Audio --- (Sin cambios relevantes aquí)
def ensure_sample_rate(waveform, original_sample_rate, desired_sample_rate=SAMPLE_RATE):
    if original_sample_rate != desired_sample_rate:
        print(f"Remuestreando audio: {original_sample_rate} Hz -> {desired_sample_rate} Hz")
        desired_length = int(round(float(len(waveform)) / original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform, desired_sample_rate

def ensure_channels(waveform, original_channels, desired_channels=1):
    if original_channels > desired_channels:
        print(f"Convirtiendo canales: {original_channels} -> {desired_channels}")
        if waveform.ndim > 1 and waveform.shape[-1] == original_channels:
            waveform = np.mean(waveform, axis=-1)
        elif waveform.ndim > 1 and waveform.shape[0] == original_channels:
             waveform = np.mean(waveform, axis=0)
        else:
             print("Advertencia: No se pudo determinar el eje de canales para la conversión a mono.")
             return waveform, original_channels
    elif original_channels < desired_channels:
         print(f"Advertencia: El audio tiene {original_channels} canales, se requieren {desired_channels}. No se puede convertir.")
         return waveform, original_channels
    return waveform, desired_channels

# --- Funciones Relacionadas con Vocabulario --- (Sin cambios)
def read_vocab(vocab_path):
    if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Archivo de vocabulario no encontrado en: {vocab_path}")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = {}
        for line in f:
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                key = parts[0]
                value = parts[1]
                vocab[key] = value
            elif len(parts) == 1:
                key = parts[0]
                vocab[key] = ""
    print(f"Vocabulario leído desde {vocab_path}, {len(vocab)} tokens encontrados.")
    return vocab

# --- Funciones de Procesamiento de Espectrograma Mel ---
def pad_or_trim(mel_array, length=MAX_LENGTH):
    original_length = mel_array.shape[1]
    if original_length > length:
        return mel_array[:, :length]
    elif original_length < length:
        pad_width = length - original_length
        # ### CAMBIO RKNNLite ### (Potencial): Considerar padding con 0 en lugar de -1.0
        # Algunos modelos cuantizados funcionan mejor con padding 0. Prueba si -1.0 da problemas.
        # return np.pad(mel_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=0.0)
        return np.pad(mel_array, ((0, 0), (0, pad_width)), mode='constant', constant_values=-1.0) # Mantener original por ahora
    else:
        return mel_array

def load_mel_filters(n_mels=N_MELS, filters_path="./model/mel_80_filters.txt"):
    if not os.path.exists(filters_path):
        raise FileNotFoundError(f"Archivo de filtros Mel no encontrado en: {filters_path}")
    try:
        expected_columns = N_FFT // 2 + 1 # 201
        mels_data = np.loadtxt(filters_path, dtype=np.float32)
        if mels_data.size != n_mels * expected_columns:
            raise ValueError(f"Tamaño inesperado de datos en {filters_path}. Se esperaban {n_mels * expected_columns} elementos, se encontraron {mels_data.size}")
        mels_data = mels_data.reshape((n_mels, expected_columns))
        print(f"Filtros Mel cargados desde {filters_path}")
        return torch.from_numpy(mels_data)
    except Exception as e:
        print(f"Error cargando o remodelando filtros Mel desde {filters_path}: {e}")
        raise

def log_mel_spectrogram(audio, n_mels=N_MELS, filters=None):
    if not torch.is_tensor(audio):
        audio = torch.from_numpy(audio)
    if filters is None:
        filters = load_mel_filters(n_mels)
    filters = filters.to(audio.device)
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True, center=True)
    # Whisper usa n_fft // 2 + 1 = 201 bins. El último puede no ser necesario para Mel.
    # La implementación original de OpenAI usa stft[:, :-1] -> 200 bins antes de .abs()**2
    # Mantengamos la consistencia con la implementación de Whisper si es posible.
    # Si tus filtros son 80x201, usa stft completo. Si son 80x200, usa stft[..., :-1]
    magnitudes = stft.abs() ** 2 # Asume filtros 80x201
    # magnitudes = stft[..., :-1].abs() ** 2 # Usar si filtros son 80x200

    mel_spec = filters @ magnitudes
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

# --- Funciones de Inferencia del Modelo ---

def run_encoder(encoder_model, mel_input):
    """Ejecuta el modelo encoder (ONNX o RKNNLite)."""
    if mel_input.shape != (1, N_MELS, MAX_LENGTH):
        raise ValueError(f"Input de Encoder con forma inesperada: {mel_input.shape}. Se esperaba (1, {N_MELS}, {MAX_LENGTH})")

    # ### CAMBIO RKNNLite ###: Usar isinstance(encoder_model, RKNNLite)
    if isinstance(encoder_model, RKNNLite):
        print("Ejecutando inferencia RKNNLite (Encoder)...")
        # ### CAMBIO RKNNLite ###: Verificar dtype esperado por el modelo RKNN
        # Podría ser np.float16, np.int8, etc. Ajusta según tu modelo.
        # Ejemplo: input_data = (mel_input * INPUT_SCALE).astype(np.float16)
        input_data = (mel_input * INPUT_SCALE).astype(np.float32) # Asumir float32 por defecto, ajustar si es necesario
        #input_data = (mel_input * INPUT_SCALE).astype(np.float16) # Cambiar a float16
        print(f"  Input shape: {input_data.shape}, dtype: {input_data.dtype}")
        outputs = encoder_model.inference(inputs=[input_data])
        out_encoder = outputs[0]
    elif isinstance(encoder_model, onnxruntime.InferenceSession):
        print("Ejecutando inferencia ONNX (Encoder)...")
        input_name = encoder_model.get_inputs()[0].name
        output_name = encoder_model.get_outputs()[0].name
        # ONNX generalmente espera float32, así que no se necesita conversión aquí
        print(f"  Input shape: {mel_input.shape}, dtype: {mel_input.dtype}")
        outputs = encoder_model.run([output_name], {input_name: mel_input})
        out_encoder = outputs[0]
    else:
        raise TypeError("Tipo de modelo Encoder no soportado. Debe ser RKNNLite u ONNX InferenceSession.")

    print(f"Salida del Encoder obtenida, forma: {out_encoder.shape}, dtype: {out_encoder.dtype}") # Esperado (1, 1500, d_model) o similar
    return out_encoder

def _decode(decoder_model, tokens, out_encoder):
    """Realiza un paso de decodificación (ONNX o RKNNLite)."""
    # tokens: es la lista de Python con la secuencia actual (longitud variable)
    # out_encoder: salida del encoder (numpy array)

    # ### CAMBIO RKNNLite ###: Usar isinstance(decoder_model, RKNNLite)
    if isinstance(decoder_model, RKNNLite):
        # print(f"Ejecutando inferencia RKNNLite (Decoder Paso)...")

        # --- INICIO: Preparar input de tokens con longitud fija ---
        current_length = len(tokens)
        # Usar un ID de padding común, como 0. Verifica si tu vocab tiene otro.
        padding_token_id = 0 # Usar <|endoftext|> en lugar de 0

        if current_length < FIXED_DECODER_INPUT_LENGTH:
            # Padear a la derecha
            pad_count = FIXED_DECODER_INPUT_LENGTH - current_length
            tokens_padded_list = tokens + [padding_token_id] * pad_count
        else:
            # Truncar (tomar los últimos N tokens)
            tokens_padded_list = tokens[-FIXED_DECODER_INPUT_LENGTH:]

        # Convertir a numpy array int64 con la forma correcta (1, FIXED_LENGTH)
        tokens_input_fixed = np.asarray([tokens_padded_list], dtype=np.int64)
        # --- FIN: Preparar input de tokens con longitud fija ---


        # Verificar dtypes esperados por el modelo RKNN Decoder
        encoder_output_typed = out_encoder.astype(np.float32) # Asumir float32, ajustar si es necesario
        # encoder_output_typed = out_encoder.astype(np.float16)

        # Imprimir formas *reales* que se pasan a inference
        # print(f"  Input tokens (fixed) shape: {tokens_input_fixed.shape}, dtype: {tokens_input_fixed.dtype}") # Debe ser (1, 12) int64
        # print(f"  Input encoder shape: {encoder_output_typed.shape}, dtype: {encoder_output_typed.dtype}")

        # RKNNLite espera una lista de inputs
        # Pasar la versión con longitud fija!
        outputs = decoder_model.inference(inputs=[tokens_input_fixed, encoder_output_typed])
        if outputs is None: # Añadir verificación por si acaso
             raise RuntimeError("La inferencia del decodificador RKNNLite devolvió None.")
        out_decoder = outputs[0] # Asumiendo que logits es la primera salida

    elif isinstance(decoder_model, onnxruntime.InferenceSession):
         # Preparar tokens_input para ONNX (puede ser variable)
         tokens_input_onnx = np.asarray([tokens], dtype=np.int64) # Shape (1, seq_len_variable)

         print(f"Ejecutando inferencia ONNX (Decoder Paso)...")
         input_names = [inp.name for inp in decoder_model.get_inputs()]
         output_names = [out.name for out in decoder_model.get_outputs()]
         print(f"  Input '{input_names[0]}' shape: {tokens_input_onnx.shape}, dtype: {tokens_input_onnx.dtype}")
         print(f"  Input '{input_names[1]}' shape: {out_encoder.shape}, dtype: {out_encoder.dtype}")
         inputs_dict = {
             input_names[0]: tokens_input_onnx,
             input_names[1]: out_encoder
         }
         outputs = decoder_model.run(output_names, inputs_dict)
         out_decoder = outputs[0]
    else:
        raise TypeError("Tipo de modelo Decoder no soportado. Debe ser RKNNLite u ONNX InferenceSession.")

    # print(f"Salida del Decoder obtenida, forma: {out_decoder.shape}, dtype: {out_decoder.dtype}")
    # La forma de salida debería seguir siendo (1, seq_len_alimentada, vocab_size)
    # Para RKNN con input fijo (1, 12), la salida debería ser (1, 12, vocab_size)
    return out_decoder


def run_decoder(decoder_model, out_encoder, vocab, task_code):
    end_token = 50257 # tokenizer.eot
    tokens = [50258, task_code, 50359, 50363] # tokenizer.sot_sequence_including_notimestamps
    timestamp_begin = 50364 # tokenizer.timestamp_begin

    max_tokens = 12
    tokens_str = ''
    pop_id = max_tokens

    tokens = tokens * int(max_tokens/4)
    next_token = 50258

    step = 0
    while next_token != end_token:
        step += 1
        out_decoder = _decode(decoder_model, tokens, out_encoder)
        next_token = out_decoder[0, -1].argmax()
        next_token_str = None
        if str(next_token) in vocab:
            next_token_str_raw = vocab[str(next_token)]

            next_token_str_processed = next_token_str_raw
            starts_with_G = next_token_str_raw.startswith('\u0120')
            core_token_str = next_token_str_raw[1:] if starts_with_G else next_token_str_raw

            corrected_core = core_token_str

            try:
                # MOJIBAKE: Convertir de latin-1 a utf-8
                original_bytes = core_token_str.encode('latin-1')
                corrected_core = original_bytes.decode('utf-8')

                if starts_with_G:
                    next_token_str_processed = '\u0120' + corrected_core
                else:
                    next_token_str_processed = corrected_core

            except (UnicodeEncodeError, UnicodeDecodeError) as e:
                 pass

            #if next_token_str_processed != next_token_str_raw:
            #     print(f"    Corrección aplicada: '{next_token_str_raw}' -> '{next_token_str_processed}'")

            next_token_str = next_token_str_processed

        else:
            # print(f"Advertencia: ID de token {next_token} no encontrado en el vocabulario.")
             next_token_str = f"<UNK_{next_token}>"

        # print(f"Paso {step}: ID Predicho={next_token}, Token='{next_token_str}'")
        tokens.append(next_token)

        if next_token == end_token:
            tokens.pop(-1)
            next_token = tokens[-1]
            break
        if next_token > timestamp_begin:
            continue
        if pop_id >4:
            pop_id -= 1

        tokens.pop(pop_id)
        tokens_str += next_token_str

    result = tokens_str.replace('\u0120', ' ').replace('<|endoftext|>', '').replace('\n', '')

    return result

# --- Funciones de Inicialización/Liberación del Modelo ---

def init_model(model_path, target=None, device_id=None):
    """Inicializa el modelo ONNX o RKNNLite."""
    print(f"Inicializando modelo desde: {model_path}")
    if model_path.endswith(".rknn"):
        # ### CAMBIO RKNNLite ###: Usar RKNNLite y init_runtime sin target/device_id
        # Crear objeto RKNNLite
        model = RKNNLite(verbose=False) # Puedes poner verbose=True para más detalles

        # Cargar modelo RKNN
        print('--> Cargando modelo RKNNLite...')
        ret = model.load_rknn(model_path)
        if ret != 0:
            print(f'Error al cargar el modelo RKNNLite "{model_path}" (código: {ret})')
            raise RuntimeError(f"Fallo al cargar RKNNLite: {ret}")
        print('    Modelo RKNNLite cargado.')

        # Inicializar entorno de ejecución en la NPU
        print('--> Inicializando entorno de ejecución RKNNLite...')
        # init_runtime para RKNNLite no suele necesitar target/device_id en el dispositivo
        ret = model.init_runtime()
        if ret != 0:
            print(f'Error al inicializar el runtime RKNNLite (código: {ret})')
            model.release() # Intentar liberar antes de salir
            raise RuntimeError(f"Fallo al inicializar runtime RKNNLite: {ret}")
        print('    Entorno RKNNLite inicializado.')
        return model

    elif model_path.endswith(".onnx"):
        print('--> Creando sesión de inferencia ONNX...')
        try:
            providers = ['CPUExecutionProvider'] # O CUDAExecutionProvider, etc. si aplica
            model = onnxruntime.InferenceSession(model_path, providers=providers)
            print(f"    Sesión ONNX creada con proveedores: {model.get_providers()}")
            return model
        except Exception as e:
            print(f"Error al crear la sesión ONNX para {model_path}: {e}")
            raise
    else:
        raise ValueError("Formato de modelo no soportado. Use .rknn o .onnx")

def release_model(model):
    """Libera los recursos del modelo."""
    # ### CAMBIO RKNNLite ###: Usar isinstance(model, RKNNLite)
    if isinstance(model, RKNNLite):
        print("Liberando modelo RKNNLite...")
        model.release()
    elif isinstance(model, onnxruntime.InferenceSession):
        print("Liberando modelo ONNX (implícito al eliminar referencia)...")
        del model # Ayuda al Garbage Collector
    else:
         print("Tipo de modelo desconocido, no se puede liberar explícitamente.")
    model = None
    print("Modelo liberado.")

# --- Función Principal ---

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Whisper Python Demo para Español (RKNNLite/ONNX)', add_help=True)
    parser.add_argument('--encoder_model_path', type=str, required=True, help='Ruta al modelo encoder (.rknn o .onnx)')
    parser.add_argument('--decoder_model_path', type=str, required=True, help='Ruta al modelo decoder (.rknn o .onnx)')
    parser.add_argument('--audio_path', type=str, required=True, help='Ruta al archivo de audio a transcribir')
    # ### CAMBIO RKNNLite ###: Target y device_id ya no son necesarios para RKNNLite en dispositivo
    # parser.add_argument('--target', type=str, default=None, help='Plataforma RKNPU destino (ej: rk3588). No necesario para RKNNLite.')
    # parser.add_argument('--device_id', type=str, default=None, help='ID del dispositivo. No necesario para RKNNLite.')
    args = parser.parse_args()

    # --- Configuración específica para Español ---
    print("\n--- Configuración para Español ---")
    vocab_path = "./model/vocab_es.txt" # Asegúrate que esta ruta es correcta
    task_code = 50262  # <|es|>
    print(f" - Vocabulario: {vocab_path}")
    print(f" - Task Code (Idioma): {task_code} (<|es|>)")
    print("----------------------------------\n")

    # --- Carga y Preprocesamiento ---
    x_mel = None # Inicializar
    try:
        print("Cargando vocabulario...")
        vocab = read_vocab(vocab_path)

        print(f"Cargando audio desde: {args.audio_path}")
        audio_data, sample_rate = sf.read(args.audio_path, dtype='float32')
        print(f"Audio original - SR: {sample_rate} Hz, Canales: {audio_data.ndim}, Duración: {len(audio_data)/sample_rate:.2f}s")

        audio_data, channels = ensure_channels(audio_data, audio_data.ndim)
        if channels != 1:
             raise ValueError("La conversión a mono falló o no fue posible.")

        audio_data, sample_rate = ensure_sample_rate(audio_data, sample_rate)
        if sample_rate != SAMPLE_RATE:
             raise ValueError("La conversión de sample rate falló.")

        print("Calculando espectrograma Log-Mel...")
        log_mel_features = log_mel_spectrogram(audio_data) # Devuelve tensor Pytorch
        log_mel_numpy = log_mel_features.numpy()

        print(f"Ajustando espectrograma a longitud {MAX_LENGTH}...")
        x_mel_padded = pad_or_trim(log_mel_numpy, length=MAX_LENGTH)

        # Añadir dimensión de batch
        x_mel = np.expand_dims(x_mel_padded, 0) # Shape final: (1, N_MELS, MAX_LENGTH)
        print(f"Preprocesamiento completado. Forma final del input: {x_mel.shape}, dtype: {x_mel.dtype}")

    except FileNotFoundError as e:
         print(f"\nError: Archivo no encontrado - {e}")
         exit(1)
    except Exception as e:
         print(f"\nError durante la carga o preprocesamiento: {e}")
         import traceback
         traceback.print_exc()
         exit(1)

    # --- Inicialización e Inferencia ---
    encoder_model = None
    decoder_model = None
    try:
        # Inicializar modelos (target y device_id ya no se pasan)
        encoder_model = init_model(args.encoder_model_path)
        decoder_model = init_model(args.decoder_model_path)

        # Ejecutar Encoder
        print("\n--- Ejecutando Encoder ---")
        out_encoder = run_encoder(encoder_model, x_mel)

        # Ejecutar Decoder
        print("\n--- Ejecutando Decoder ---")
        result = run_decoder(decoder_model, out_encoder, vocab, task_code)

        print("\n--- Transcripción Resultante ---")
        print(result)
        print("------------------------------\n")

    except Exception as e:
        print(f"\nError durante la inicialización o inferencia: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # --- Liberar Recursos ---
        if encoder_model:
            release_model(encoder_model)
        if decoder_model:
            release_model(decoder_model)
        print("Proceso finalizado.")
