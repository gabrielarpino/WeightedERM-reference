"""Keep regenerated final notebook sources aligned with the paper figure style."""

import re


def normalize_presentation(source):
    """Set 18 pt axis labels, 14 pt ticks, and at most three displayed digits."""
    source = re.sub(r"(['\"]font\.size['\"]\s*:\s*)10\b", r"\g<1>14", source)
    if "axes.labelsize" not in source and "'font.size': 14," in source:
        source = source.replace(
            "'font.size': 14,",
            "'font.size': 14,\n    'axes.labelsize': 18, "
            "'axes.titlesize': 18, 'legend.fontsize': 14,",
        )
    source = re.sub(r"(['\"]axes\.labelsize['\"]\s*:\s*)16\b", r"\g<1>18", source)
    source = re.sub(r"(['\"](?:xtick|ytick)\.labelsize['\"]\s*:\s*)9\b", r"\g<1>14", source)
    source = re.sub(r"\bx_y_labels?_fontsize\s*=\s*22\b", lambda m: m.group().replace("22", "18"), source)
    source = re.sub(r"(set_[xy]label\([^\n]*?fontsize=)(?:16|22)(?=[,)])", r"\g<1>18", source)
    source = re.sub(r"(set_[xy]ticklabels\([^\n]*?fontsize=)(?:12|18|22|x_y_label_fontsize)(?=[,)])", r"\g<1>14", source)
    source = re.sub(r"(tick_params\([^\n]*?labelsize=)(?:18|22|x_y_label_fontsize|x_y_labels_fontsize)(?=[,)])", r"\g<1>14", source)
    source = re.sub(r"(set_[xy]label\([^\n]*?fontsize=)x_y_labels?_fontsize(?=[,)])", r"\g<1>18", source)
    source = re.sub(r"(?<=\.)\d+(?=[fg])", lambda m: "3" if int(m.group()) > 3 else m.group(), source)
    source = re.sub(r"(?<=precision=)\d+", lambda m: "3" if int(m.group()) > 3 else m.group(), source)
    return source
