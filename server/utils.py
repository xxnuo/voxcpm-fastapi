import re


def normalize_str_to_safe_filename(s: str) -> str:
    """
    将字符串标准化为跨平台（Linux, macOS, Windows）安全的文件名。

    处理：
    1. 移除或替换 Windows 和 POSIX 文件系统中的非法字符。
    2. 移除或替换空字符 (\0)。
    3. 处理文件名过长问题（截断）。
    4. 确保文件名不是 Windows 保留名称。

    :param s: 待标准化的原始字符串。
    :return: 标准化后的安全文件名。
    """

    illegal_chars = r'[\\/:*?"<>|\0]'

    safe_name = re.sub(illegal_chars, "_", s)

    safe_name = safe_name.strip()
    safe_name = safe_name.rstrip(".")

    windows_reserved_names = r"^(con|prn|aux|nul|com\d|lpt\d)$"

    if re.fullmatch(windows_reserved_names, safe_name, re.IGNORECASE):
        safe_name = f"safe_{safe_name}"

    MAX_FILENAME_LENGTH = 200
    if len(safe_name) > MAX_FILENAME_LENGTH:
        safe_name = safe_name[:MAX_FILENAME_LENGTH]

    if not safe_name:
        return "empty_filename"

    return safe_name
