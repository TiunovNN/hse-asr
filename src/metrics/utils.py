from jiwer import wer, cer


def calc_cer(target_text, predicted_text) -> float:
    if not target_text:
        return int(len(predicted_text) > 0)

    return cer([target_text], [predicted_text])


def calc_wer(target_text, predicted_text) -> float:
    if not target_text:
        return int(len(predicted_text) > 0)

    return wer([target_text], [predicted_text])
