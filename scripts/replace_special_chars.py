# df = pd.read_parquet(current_dir / './midjourney-prompts/data/validation-00000-of-00001.parquet')


def replace_special_chars(str):
    return str.replace("--", "").replace(":", '').replace("+", '').replace(",", '').replace(".", '').replace("""\"""", '').replace("'", '').replace("!", '').replace("?", '').replace(";", '').replace("(", '').replace(")", '').replace("/", '').replace("\\", '').replace("<", "").replace('>', '')