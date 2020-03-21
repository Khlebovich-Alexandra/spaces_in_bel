import pandas as pd

# from polyglot.detect import Detector

text_table = pd.read_csv('Data/table_lang.csv', index_col=0)
text_table.lang_list.fillna('', inplace=True)
text_table.lang_list = text_table.lang_list.map(lambda x: x.strip('][').split(', '))

text_table = text_table.dropna()
text_table.reset_index(inplace=True, drop=True)

text_table['length'] = text_table.text.map(len)
text_table['is_bel_program'] = text_table.lang_list.map(lambda x: "'Belarusian'" in x)
text_table['was_na'] = text_table.lang_list.map(lambda x: '' in x)
text_table['with_y_neskl'] = text_table.text.map(lambda x: 'ў' in x.lower())
text_table['nnsrc'] = text_table.text.map(lambda x: 'nnsrc' in x)
text_table['belsat_eu'] = text_table.text.map(lambda x: 'belsat.eu' in x)

text_table['is_bel'] = text_table.is_bel_program | text_table.with_y_neskl | text_table.nnsrc

ind = text_table[text_table.was_na & ~text_table.is_bel & (text_table.length < 36)].index
text_table.at[ind, 'is_bel'] = True

text = '\n'.join(text_table[text_table.is_bel].text)
with open('Data/train-bel-clear.txt', 'w', encoding='utf-8') as fileObject:
    fileObject.write(text)
