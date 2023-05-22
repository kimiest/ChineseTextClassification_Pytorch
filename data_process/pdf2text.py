import os
from io import StringIO
import json
from tqdm import tqdm

from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def convert_pdf_2_text(path):
    '''
    将单个pdf文件转换为text
    :param path:
    :return:
    '''
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    device = TextConverter(rsrcmgr, retstr, codec='utf-8', laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    with open(path, 'rb') as fp:
        for page in PDFPage.get_pages(fp, set()):
            interpreter.process_page(page)
        text = retstr.getvalue()
        device.close()
        retstr.close()
        return text.replace(' ', '').replace('\n', '').replace('\f', '')

def convert_batch(root, type='unknown'):
    '''
    将指定文件夹中的一批PDF全部转换为text
    :param root:
    :param type:
    :return:
    '''
    data = []
    for file in tqdm(os.listdir(root)):
        cur_pdf = os.path.join(root, file)
        cur_text = convert_pdf_2_text(cur_pdf)
        data.append({'uid': file.strip('.pdf'), 'text': cur_text, 'type': type})
    return data


if __name__ == '__main__':
    '''将eval数据全部转换为text，保存为.json格式'''
    # with open(r'E:\Code\FinancialReportingTyping\data\eval_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(eval_data, f, ensure_ascii=False)

    '''将train数据全部转换为text，保存为.json格式'''
    batch_0 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\晨会早报-new', type=0)
    batch_1 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\宏观研报-new', type=1)
    batch_2 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\策略研报-new', type=2)
    batch_3 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\行业研报-new', type=3)
    batch_4 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\公司研报-new', type=4)
    batch_5 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\基金研报-new', type=5)
    batch_6 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\债券研报-new', type=6)
    batch_7 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\金融工程-new', type=7)
    batch_8 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\其他研报-new', type=8)
    batch_9 = convert_batch(r'E:\Code\FinancialReportingTyping\data\train_dataset\个股研报-new', type=9)
    with open(r'E:\Code\FinancialReportingTyping\data\train_data.json', 'w', encoding='utf-8') as f:
        json.dump(batch_0+batch_1+batch_2+batch_3+batch_4+batch_5+batch_6+batch_7+batch_8+batch_9,
                  f, ensure_ascii=False)

