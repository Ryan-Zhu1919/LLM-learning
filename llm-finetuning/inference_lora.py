from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 1. 配置模型路径
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_DIR = (
    Path(__file__).resolve().parent
    / "qwen2_5_0_5b_lora_adapter"
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("使用设备：", device)

# 2. 加载 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    str(ADAPTER_DIR),
    local_files_only=True
)

# 3. 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    local_files_only=True
)

# 4. 加载训练好的 LoRA Adapter
model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR),
    is_trainable=False
)

model = model.to(device)
model.eval()

print("模型加载完成。")

# 5. 定义生成回答的函数
def generate_answer(question: str) -> str:
    messages = [
        {"role": "user", "content": question}
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=(
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            eos_token_id=tokenizer.eos_token_id
        )

    # 只提取新生成的 Token
    new_tokens = output_ids[
        0, inputs["input_ids"].shape[1]:
    ]
    # 将 Token ID 转换成文本
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True
    )

    return answer.strip()

# 6. 交互式问答
while True:
    question = input("\n你：").strip()
    if question.lower() in {"q", "quit", "exit"}:
        print("程序结束。")
        break
    if not question:
        continue

    answer = generate_answer(question)
    print("模型：", answer)