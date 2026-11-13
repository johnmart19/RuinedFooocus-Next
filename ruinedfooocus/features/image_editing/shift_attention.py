import re
import math

# Take prompts with "(token:start value~end value)" and let the weight change according to "distance"
# From: https://github.com/yownas/shift-attention

_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
re_attention_span = re.compile(r"(?<!\\)(\([^()]*:)(" + _NUMBER + r"(?:~" + _NUMBER + r")+)(\))")


def shift_attention(text, distance):

    def inject_value(distance, match_obj):
        a = match_obj.group(2).split("~")
        l = len(a) - 1
        q1 = int(math.floor(distance * l))
        q2 = int(math.ceil(distance * l))
        value = float(a[q1]) + ((float(a[q2]) - float(a[q1])) * (distance * l - q1))
        return match_obj.group(1) + str(value) + match_obj.group(3)

    res = re.sub(re_attention_span, lambda match_obj: inject_value(distance, match_obj), text)
    return res
