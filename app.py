"""Interactive command-line application for BERT word similarity."""

from word_similarity import BertWordSimilarity


def main() -> None:
    print("正在加载 bert-base-uncased，首次运行需要下载模型……")
    similarity = BertWordSimilarity()
    print("模型加载完成。输入 q 可退出。\n")

    while True:
        candidate1 = input("候选单词 1: ").strip()
        if candidate1.lower() == "q":
            break

        candidate2 = input("候选单词 2: ").strip()
        if candidate2.lower() == "q":
            break

        new_word = input("新单词: ").strip()
        if new_word.lower() == "q":
            break

        try:
            result = similarity.compare(candidate1, candidate2, new_word)
        except ValueError as exc:
            print(f"输入错误: {exc}\n")
            continue

        print(f"最接近的候选单词: {result.closest_word}")
        print(
            f"相似度: {candidate1}={result.candidate1_score:.4f}, "
            f"{candidate2}={result.candidate2_score:.4f}\n"
        )

    print("程序已退出。")


if __name__ == "__main__":
    main()

