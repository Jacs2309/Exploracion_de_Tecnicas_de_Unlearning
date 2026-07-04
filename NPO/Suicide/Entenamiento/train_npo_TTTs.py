"""
Script de Entrenamiento de Desaprendizaje (Machine Unlearning) usando NPO
Optimizado para Máquinas Virtuales de Google Cloud Platform (GCP)
Modelo Base: Qwen/Qwen3-14B (o Qwen/Qwen2.5-14B-Instruct)
"""

import os
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download, login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# ==========================================
# 1. CONFIGURACIÓN DE CREDENCIALES Y RUTA (GCP)
# ==========================================
# REQUISITO CRÍTICO: Reemplaza con tu token real con permisos de ESCRITURA (Write)
HF_TOKEN = os.getenv("HF_TOKEN")

# Nombre de usuario de Hugging Face para crear el repositorio privado automáticamente
HF_USER = "CarpaMagica"

# Configuración de rutas y modelo
MODEL_NAME = "Qwen/Qwen3-14B"  # Cambiar a "Qwen/Qwen2.5-14B-Instruct" si se prefiere estabilidad sin modo thinking
OUTPUT_DIR = "./npo-qwen-TTTs-7"
DATASET_PATH = "minimax_TTTs_v9.jsonl"  # Asegúrate de subir este archivo al mismo directorio

# Inicializar autenticación en Hugging Face
print("Autenticando en Hugging Face...")
login(token=os.environ["HF_TOKEN"], add_to_git_credential=True)

# Hiperparámetros de LoRA y Entrenamiento
LORA_R = 32
LORA_ALPHA = 64
LEARNING_RATE = 5e-6
MAX_LENGTH = 512
BETA = 0.1

# Configuración de LoRA extendida a capas MLP para potenciar la lógica inversa del juego
LORA_CONFIG = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# ==========================================
# 2. PREPARACIÓN DE DATOS (PROMPT OPTIMIZADO)
# ==========================================
def prepare_dataset(jsonl_path, tokenizer):
    from datasets import load_dataset
    print(f"Cargando dataset desde {jsonl_path}...")
    dataset = load_dataset("json", data_files=jsonl_path, split="train")

    def format_dpo_row(x):
        return {
            "prompt":   str(x["prompt"]),
            "chosen":   str(x["chosen"])   + tokenizer.eos_token,
            "rejected": str(x["rejected"]) + tokenizer.eos_token,
        }

    return dataset.map(format_dpo_row)

# ==========================================
# 3. PERSONALIZACIÓN DEL TRAINER NPO
# ==========================================
class NPOTrainer(DPOTrainer):
    def dpo_loss(
        self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        *args,
        **kwargs
    ):
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        losses = -(2.0 / self.beta) * F.logsigmoid(-self.beta * rejected_logratios)
        chosen_rewards   = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * rejected_logratios.detach()
        return losses, chosen_rewards, rejected_rewards

    def log(self, logs: dict, *args, **kwargs):
        """
        Atrapa el argumento extra (start_time) que manda la nueva versión 
        de Transformers y evita que rompa la versión de TRL.
        """
        try:
            super().log(logs, *args, **kwargs)
        except TypeError:
            super().log(logs)

# ==========================================
# 4. PIPELINE PRINCIPAL DE ENTRENAMIENTO
# ==========================================
def train_npo():
    print(f"Descargando snapshot completo de {MODEL_NAME} de forma limpia y segura...")
    # snapshot_download evita problemas de corrupción de shards intermitentes en la red
    local_model_dir = snapshot_download(
        repo_id=MODEL_NAME,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.msgpack", "*.h5"]
    )
    print(f"Modelo base descargado localmente en: {local_model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    tokenizer.pad_token = tokenizer.eos_token

    print("Cargando modelo en bfloat16 nativo...")
    model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    # Preparación de Peft con LoRA completo
    model = get_peft_model(model, LORA_CONFIG)
    # Fix para gradient checkpointing + LoRA + multi-GPU:
    # enable_input_require_grads() no es suficiente con device_map="auto"
    # El hook manual garantiza requires_grad=True en las activaciones de entrada
    def make_inputs_require_grad(module, input, output):
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        elif isinstance(output, tuple):
            for o in output:
                if isinstance(o, torch.Tensor):
                    o.requires_grad_(True)
 
    # Registrar en el embedding layer — primer módulo del forward pass
    model.base_model.model.model.embed_tokens.register_forward_hook(make_inputs_require_grad)
 
    model.print_trainable_parameters()

    # Configuración limpia de DPO/NPO
    training_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        max_length=MAX_LENGTH,
        num_train_epochs=3,
        beta=BETA,
        loss_type="sigmoid",
        report_to="none",
	    logging_steps=5,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        remove_unused_columns=False,
        warmup_ratio=0.1,
        max_steps = 40, 
        save_steps = 10,
        save_total_limit = 4,
    )

    dataset = prepare_dataset(DATASET_PATH, tokenizer)

    trainer = NPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("¡Comenzando el entrenamiento NPO de Desaprendizaje!")
    trainer.train()
    
    print(f"Guardando adaptador localmente en: {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
# ==========================================
# 5. SUBIDA AUTOMÁTICA A HUGGING FACE PRIVADO
# ==========================================
    try:
        repo_name = "npo-qwen-TTTs-7"
        repo_id = f"{HF_USER}/{repo_name}"
        api = HfApi()
        
        print(f"Creando repositorio privado en el Hub: {repo_id}...")
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=True,
            exist_ok=True
        )
        
        print(f"Subiendo archivos del adaptador LoRA de forma segura...")
        api.upload_folder(
            folder_path=OUTPUT_DIR,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Subida automática del adaptador de desaprendizaje NPO desde GCP VM"
        )
        print(f"¡Éxito total! Tu adaptador está respaldado de forma privada en: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Alerta al subir al Hub (los archivos locales se guardaron bien): {e}")

    return model, tokenizer

if __name__ == "__main__":
    train_npo()

