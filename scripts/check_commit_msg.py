import sys


ERROR_CODE = 1


def check_commit_message(
    path_to_file_with_message: str,
    min_words_amount: int = 2,
    min_message_len: int = 7,
) -> None:
    with open(path_to_file_with_message) as file:
        commit_msg = file.read().strip()

    if len(commit_msg) < min_message_len:
        print(f"commit message must contain at least {min_message_len} characters")
        sys.exit(ERROR_CODE)

    if not commit_msg[0].isupper():
        print(
            "commit message must start with uppercase letter "
            f"but the next commit message was give: {commit_msg}"
        )
        sys.exit(ERROR_CODE)

    if (words_amount := len(commit_msg.split())) < min_words_amount:
        print(
            f"commit message must contain at least {min_words_amount}"
            f" words but the next amount {words_amount} is used"
        )
        sys.exit(ERROR_CODE)


if __name__ == "__main__":
    check_commit_message(sys.argv[1])
    exit(0)
