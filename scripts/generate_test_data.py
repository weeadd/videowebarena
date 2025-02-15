"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json
import os

from browser_env.env_config import *


def main() -> None:
    DATASET = os.environ["DATASET"]
    if DATASET == "webarena":
        print("DATASET: webarena")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
        print(f"GITLAB: {GITLAB}")
        print(f"WIKIPEDIA: {WIKIPEDIA}")
        print(f"MAP: {MAP}")
        
        inp_paths = ["config_files/wa/test_webarena.raw.json"]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__GITLAB__": GITLAB,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__MAP__": MAP,
        }
    elif DATASET == "visualwebarena":
        print("DATASET: visualwebarena")
        print(f"CLASSIFIEDS: {CLASSIFIEDS}")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        inp_paths = [
            "config_files/vwa/test_classifieds.raw.json", "config_files/vwa/test_shopping.raw.json", "config_files/vwa/test_reddit.raw.json"
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__CLASSIFIEDS__": CLASSIFIEDS,
        }
    elif DATASET == "videowebarena":
        print("DATASET: visualwebarena")
        print(f"CLASSIFIEDS: {CLASSIFIEDS}")
        print(f"REDDIT: {REDDIT}")
        print(f"SHOPPING: {SHOPPING}")
        print(f"SHOPPING_ADMIN: {SHOPPING_ADMIN}")
        print(f"GITLAB: {GITLAB}")
        print(f"MAP: {MAP}")
        inp_paths = [
           "config_files/videowa/test_classifieds.raw.json", "config_files/videowa/test_shopping.raw.json", "config_files/videowa/test_reddit.raw.json", "config_files/videowa/test_shopping_admin.raw.json",  "config_files/videowa/test_map.raw.json",  "config_files/videowa/test_gitlab.raw.json", 
        ]
        replace_map = {
            "__REDDIT__": REDDIT,
            "__SHOPPING__": SHOPPING,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__CLASSIFIEDS__": CLASSIFIEDS,
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__GITLAB__": GITLAB,
            "__MAP__": MAP,
        }    
    else:
        raise ValueError(f"Dataset not implemented: {DATASET}")
        
    for inp_path in inp_paths:
        output_dir = inp_path.replace('.raw.json', '')
        os.makedirs(output_dir, exist_ok=True)
        with open(inp_path, "r") as f:
            raw = f.read()
        for k, v in replace_map.items():
            raw = raw.replace(k, v)

        with open(inp_path.replace(".raw", ""), "w") as f:
            f.write(raw)
        data = json.loads(raw)
        for idx, item in enumerate(data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                json.dump(item, f, indent=2)
        print("FINISHED", inp_path)


if __name__ == "__main__":
    main()
