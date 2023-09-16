import eng_to_ipa as ipa
import pykakasi

def main():
    eng_to_ipa = {"a": "ə", "ey": "eɪ", "aa": "ɑ", "ae": "æ", "ah": "ə", "ao": "ɔ",
               "aw": "aʊ", "ay": "aɪ", "ch": "ʧ", "dh": "ð", "eh": "ɛ", "er": "ər",
               "hh": "h", "ih": "ɪ", "jh": "ʤ", "ng": "ŋ",  "ow": "oʊ", "oy": "ɔɪ",
               "sh": "ʃ", "th": "θ", "uh": "ʊ", "uw": "u", "zh": "ʒ", "iy": "i", "y": "j"}
    ipa_to_eng = dict([[v, k] for k, v in eng_to_ipa.items()])
    breakpoint()
    
    text_en = 'The quick brown fox jumped over the lazy dog.'
    phoneme_en = ipa.convert(text_en, keep_punct=True)
    print(phoneme_en)

    kakasi = pykakasi.kakasi()
    text_ja = '泥棒でもはいったかと一瞬僕は思った。'
    text_ja_roma = kakasi.convert(text_ja)
    text_ja_roma = [t['passport'] for t in text_ja_roma if t ['passport'] != ' ']
    text_ja_roma = ' '.join(text_ja_roma)
    print(text_ja_roma)

    phoneme_ja = ipa.convert(text_ja_roma, keep_punct=True)
    phoneme_ja = phoneme_ja.replace('*', '')
    print(phoneme_ja)


if __name__ == '__main__':
    main()
