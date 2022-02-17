from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--test", action="store_true")
parser.add_argument("-e", "--export", default=None, type=Path)
args = parser.parse_args()

train_langs = ['hy', 'cy', 'no', 'orv', 'en', 'fr', 'qhe', 'sl', 'kmr', 'it', 'tr', 'fi', 'id', 'uk', 'nl', 'pl', 'pt', 'kk', 'la', 'fro', 'es', 'bxr', 'ko', 'et', 'hr', 'got', 'swl', 'sme', 'pcm', 'de', 'lv', 'zh', 'lt', 'gl', 'vi', 'el', 'ca', 'sv', 'ru', 'cs', 'mr', 'eu', 'sk', 'ta', 'mt', 'grc', 'is', 'ur', 'ro', 'fa', 'ja', 'hu', 'hi', 'lzh', 'fo', 'sa', 'olo', 'ar', 'wo', 'bg', 'te', 'qtd', 'cu', 'hsb', 'da', 'ga', 'af', 'be', 'cop', 'sr', 'hyw', 'gd', 'he', 'ug']

test_langs = ['akk', 'hy', 'cy', 'no', 'orv', 'en', 'sq', 'fr', 'qhe', 'sl', 'gub', 'kmr', 'it', 'tr', 'fi', 'id', 'uk', 'nl', 'pl', 'pt', 'kk', 'la', 'fro', 'es', 'bxr', 'urb', 'ko', 'et', 'hr', 'got', 'swl', 'gsw', 'aii', 'sme', 'pcm', 'de', 'lv', 'zh', 'tl', 'bm', 'lt', 'gl', 'vi', 'am', 'el', 'ca', 'soj', 'sv', 'ess', 'ru', 'cs', 'bej', 'myv', 'bho', 'th', 'mr', 'eu', 'sk', 'quc', 'yo', 'wbp', 'nds', 'ta', 'mt', 'grc', 'is', 'gun', 'ur', 'ro', 'fa', 'apu', 'ja', 'hu', 'hi', 'lzh', 'koi', 'fo', 'sa', 'olo', 'ar', 'wo', 'bg', 'aqz', 'mpu', 'xnr', 'br', 'te', 'yue', 'qtd', 'cu', 'krl', 'hsb', 'da', 'ajp', 'kpv', 'ga', 'nyq', 'qfn', 'myu', 'gv', 'sms', 'af', 'otk', 'tpn', 'be', 'cop', 'sr', 'mdf', 'hyw', 'gd', 'kfm', 'he', 'ug', 'ckt']

lang_names = {'akk': 'Akkadian', 'hy': 'Armenian', 'cy': 'Welsh', 'no': 'Norwegian', 'orv': 'Old_East_Slavic', 'en': 'English', 'sq': 'Albanian', 'fr': 'French', 'qhe': 'Hindi_English', 'sl': 'Slovenian', 'gub': 'Guajajara', 'kmr': 'Kurmanji', 'it': 'Italian', 'tr': 'Turkish', 'fi': 'Finnish', 'id': 'Indonesian', 'uk': 'Ukrainian', 'nl': 'Dutch', 'pl': 'Polish', 'pt': 'Portuguese', 'kk': 'Kazakh', 'la': 'Latin', 'fro': 'Old_French', 'es': 'Spanish', 'bxr': 'Buryat', 'urb': 'Kaapor', 'ko': 'Korean', 'et': 'Estonian', 'hr': 'Croatian', 'got': 'Gothic', 'swl': 'Swedish_Sign_Language', 'gsw': 'Swiss_German', 'aii': 'Assyrian', 'sme': 'North_Sami', 'pcm': 'Naija', 'de': 'German', 'lv': 'Latvian', 'zh': 'Chinese', 'tl': 'Tagalog', 'bm': 'Bambara', 'lt': 'Lithuanian', 'gl': 'Galician', 'vi': 'Vietnamese', 'am': 'Amharic', 'el': 'Greek', 'ca': 'Catalan', 'soj': 'Soi', 'sv': 'Swedish', 'ess': 'Yupik', 'ru': 'Russian', 'cs': 'Czech', 'bej': 'Beja', 'myv': 'Erzya', 'bho': 'Bhojpuri', 'th': 'Thai', 'mr': 'Marathi', 'eu': 'Basque', 'sk': 'Slovak', 'quc': 'Kiche', 'yo': 'Yoruba', 'wbp': 'Warlpiri', 'nds': 'Low_Saxon', 'ta': 'Tamil', 'mt': 'Maltese', 'grc': 'Ancient_Greek', 'is': 'Icelandic', 'gun': 'Mbya_Guarani', 'ur': 'Urdu', 'ro': 'Romanian', 'fa': 'Persian', 'apu': 'Apurina', 'ja': 'Japanese', 'hu': 'Hungarian', 'hi': 'Hindi', 'lzh': 'Classical_Chinese', 'koi': 'Komi_Permyak', 'fo': 'Faroese', 'sa': 'Sanskrit', 'olo': 'Livvi', 'ar': 'Arabic', 'wo': 'Wolof', 'bg': 'Bulgarian', 'aqz': 'Akuntsu', 'mpu': 'Makurap', 'xnr': 'Kangri', 'br': 'Breton', 'te': 'Telugu', 'yue': 'Cantonese', 'qtd': 'Turkish_German', 'cu': 'Old_Church_Slavonic', 'krl': 'Karelian', 'hsb': 'Upper_Sorbian', 'da': 'Danish', 'ajp': 'South_Levantine_Arabic', 'kpv': 'Komi_Zyrian', 'ga': 'Irish', 'nyq': 'Nayini', 'qfn': 'Frisian_Dutch', 'myu': 'Munduruku', 'gv': 'Manx', 'sms': 'Skolt_Sami', 'af': 'Afrikaans', 'otk': 'Old_Turkish', 'tpn': 'Tupi', 'be': 'Belarusian', 'cop': 'Coptic', 'sr': 'Serbian', 'mdf': 'Moksha', 'hyw': 'Western_Armenian', 'gd': 'Scottish_Gaelic', 'kfm': 'Khunsari', 'he': 'Hebrew', 'ug': 'Uyghur', 'ckt': 'Chukchi'}



sizes = []
for langid in (test_langs if args.test else train_langs):
    d = load_dataset("./datasets/udpos28", langid, split="test" if args.test else "train")
    sizes.append((langid, d.num_rows))

df = pd.DataFrame(sizes, columns=["language", "size"])
print(df)

if args.export:
    df.to_csv(args.export)
