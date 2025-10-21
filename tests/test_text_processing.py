from ray.text_processing import split_text


def test_split_text_handles_chinese_punctuation():
    text = """
    他抬头望着天空，轻声道：“今晚会下雨吗？”
    她摇摇头，笑着回答：“不会的！”
    黎明时分，街道上已经被晨雾笼罩。
    """.strip()
    chunks = split_text(text, max_length=20, overlap=4)
    assert len(chunks) >= 3
    # 句号和问号后应被切分
    assert any("今晚会下雨吗" in chunk for chunk in chunks)
    assert any("不会的" in chunk for chunk in chunks)
