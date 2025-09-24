import os
from hunyuan_mt_7B_model import model, tokenizer
import argparse
import sys

#region How to Run
'''
python3 translate_batch.py \
    --input <INPUT_TXT_FILE_PATH> \
    --output <OUTPUT_TXT_FILE_PATH> \
    --source <SOURCE_LANG> \
    --target <TARGET_LANG>
    '''
#endregion

#region Translation Helpers

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translate text from `source_lang` to `target_lang` using Hunyuan-MT model.
    """
    # Build prompt dynamically
    if "Chinese" in (source_lang, target_lang):
        prompt = f"把下面的文本翻译成{target_lang}，不要额外解释。\n\n{text}"
    else:
        prompt = f"Translate the following segment into {target_lang}, without additional explanation.\n\n{text}"

    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        inputs,
        max_new_tokens=2048,
        top_k=20,
        top_p=0.6,
        repetition_penalty=1.05,
        temperature=0.7
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    translated = decoded.split("<|extra_0|>", 1)[1] if "<|extra_0|>" in decoded else decoded

    for token in ["<|startoftext|>", "<|eos|>"]:
        translated = translated.replace(token, "")

    return translated.strip()

#endregion

#region Batch File Translation

def translate_file(input_path: str, output_path: str, source_lang: str, target_lang: str):
    """
    Translate each line from the input file and write results to the output file.
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for idx, line in enumerate(fin, start=1):
            text = line.strip()
            if not text:
                fout.write("\n")
                continue

            try:
                translated = translate_text(text, source_lang, target_lang)
            except Exception as e:
                translated = f"[ERROR on line {idx}: {e}]"
                print(f"Error translating line {idx}: {e}")

            fout.write(translated + "\n")
            print(f"[Line {idx}] Translated: {translated}")

    print(f"Translations completed. Output saved to: {output_path}")

#endregion

def validate_args(args):
    if not os.path.isfile(args.input):
        raise ValueError(f"Input file does not exist: {args.input}")

    if not args.output.strip():
        raise ValueError("Output path cannot be empty.")

    if not args.source.strip():
        raise ValueError("Source language cannot be empty.")

    if not args.target.strip():
        raise ValueError("Target language cannot be empty.")

def main():
    parser = argparse.ArgumentParser(description="Batch translate text file using Hunyuan-MT model.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input text file.")
    parser.add_argument("--output", "-o", required=True, help="Path for saving the translated output.")
    parser.add_argument("--source", "-s", required=True, help="Source language name (e.g., English, Chinese).")
    parser.add_argument("--target", "-t", required=True, help="Target language name (e.g., English, Portuguese).")

    args = parser.parse_args()

    try:
        validate_args(args)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

    translate_file(args.input, args.output, args.source, args.target)


if __name__ == "__main__":
    main()
