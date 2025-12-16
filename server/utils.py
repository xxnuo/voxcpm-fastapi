def normalize_str_to_safe_filename(s: str) -> str:
    """
    将字符串标准化为安全的文件名，仅针对 Linux (POSIX)。
    替换'/'和'\0'为下划线，其余字符允许。
    """
    invalid_chars = set("/")
    return "".join((c if c not in invalid_chars and c != "\0" else "_") for c in s)
