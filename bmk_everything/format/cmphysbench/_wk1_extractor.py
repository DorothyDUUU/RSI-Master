import re
GT=chr(62); LT=chr(60); LP=chr(40); RP=chr(41); BS=chr(92); OB=chr(123); CB=chr(125); DOT=chr(46); STAR=chr(42); QM=chr(63)
def _tag_all(tag, text):
    inner = LP + DOT + STAR + QM + RP
    pat = LT + tag + chr(91) + chr(94) + GT + chr(93) + chr(42) + GT + inner + LT + chr(47) + tag + GT
    return re.findall(pat, text, re.DOTALL | re.IGNORECASE)
def _strip_tags(text):
    pat = LT + chr(47) + QM + chr(91) + chr(94) + GT + chr(93) + chr(43) + GT
    return re.sub(pat, chr(32), text)
def balanced_boxed_spans(text):
    spans = []
    tok = BS + "boxed" + OB
    idx = 0
    while True:
        s = text.find(tok, idx)
        if s < 0: break
        cstart = s + len(tok); d = 1; i = cstart
        while i < len(text):
            c = text[i]
            if c == OB: d += 1
            elif c == CB:
                d -= 1
                if d == 0: break
            i += 1
        if d == 0:
            spans.append((s, i + 1, text[cstart:i].strip())); idx = i + 1
        else:
            break
    return spans
def extract_format(response):
    if not response: return None
    qs = _tag_all("question", response)
    ss = _tag_all("solution", response)
    if len(qs) != 1: return None
    question = qs[0].strip()
    if len(question) < 30: return None
    if len(ss) >= 1:
        body = ss[0]
        after = response.split(ss[0], 1)[1] if ss[0] in response else ""
    else:
        body = response
        after = ""
    region = body + (chr(32) + after if after else "")
    spans = balanced_boxed_spans(region)
    if len(spans) != 1: return None
    bs, be, boxed = spans[0]
    if not boxed: return None
    if boxed.count(OB) != boxed.count(CB): return None
    NL = chr(10)
    thinks = _tag_all("think", body)
    if len(thinks) == 1 and thinks[0].strip():
        think = thinks[0].strip()
    else:
        deriv = _strip_tags(region[:bs]).strip()
        if len(deriv) < 40: return None
        think = deriv
    assistant = LT + "think" + GT + think + LT + chr(47) + "think" + GT + NL + BS + "boxed" + OB + boxed + CB
    return {"question": question, "think": think, "boxed": boxed, "assistant": assistant}
