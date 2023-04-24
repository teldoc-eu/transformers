import onnxruntime as rt
import numpy as np

from transformers import BartTokenizer, AutoConfig, BartTokenizerFast

ARTICLE_TO_SUMMARIZE_LIST = [['Informacje o pacjencie\n\n\n0\nORGUNITNAME\nOrt Oddział Ortopedii i Traumatologii\n\n\n1\nAGE\n68\n\n\n3\nSEX\nF\n\n\n4\nCITIZENSHIP\nPL\n\n\n5\nPLACEOFBIRTH\nStoczek Łukowski\n\n\n6\nPOSTALCODE\n05-803\n\n\n7\nFULLNAME\nPRUSZKÓW - GMINA MIEJSKA\n\n\n', 'Epizod Szpitalny, 2007-06-20\n20:51:00Koniec epizodu: 2007-06-22 15:11:00Event IP - Izba PrzyjęćPoczątek eventu2007-06-20 20:51:00Koniec eventu2007-06-20 20:52:00Event SOR - Szpitalny\nOddział RatunkowyPoczątek eventu2007-06-20 20:52:00Koniec eventu2007-06-20 23:01:00ICD10\n\n\nS60\nPowierzchowny uraz nadgarstka i\nręki\n\n\nMAINICD10\n\n\nS60\nPowierzchowny uraz nadgarstka i\nręki\n\n\nEvent Ort -\nOddział Ortopedii i TraumatologiiPoczątek eventu2007-06-20 23:01:00Koniec eventu2007-06-22 15:11:00ICD10\n\n\n\n\n\n\nS52\nZłamanie\nprzedramienia\n\n\nW01.1\nUPADEK NA TYM SAMYM POZIOMIE\nWSKUTEK POTKNIECIA, POSLIZGNIECIA - INSTYTUCJA MIESZKALNA\n\n\nMAINICD10\n\n\nS52\nZłamanie\nprzedramienia\n\n\n', 'Epizod Ambulatoryjny,\n2007-06-22 15:04:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-08-01 08:43:00Koniec eventu2007-08-01 09:03:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-08-08 13:55:00Koniec eventu2007-08-08 14:15:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-08-30 09:58:00Koniec eventu2007-08-30 10:18:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-09-27 10:42:00Koniec eventu2007-09-27 11:02:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-11-19 15:45:00Koniec eventu2007-11-19 16:05:00Event 1007 -\nGabinet 1007 Poradni OrtopedycznejPoczątek eventu2008-06-20 09:15:00Koniec eventu2008-06-20 09:30:00', 'Epizod Szpitalny, 2007-08-21\n09:12:00Koniec epizodu: 2007-08-24 13:20:00Event IP - Izba PrzyjęćPoczątek eventu2007-08-21 09:12:00Koniec eventu2007-08-21 09:13:00ICD10\n\n\nS62\nZłamanie na poziomie nadgarstka i\nręki\n\n\nEvent Ort -\nOddział Ortopedii i TraumatologiiPoczątek eventu2007-08-21 09:13:00Koniec eventu2007-08-24 13:20:00ICD10\n\n\n\n\n\n\nS52.2\nZłamanie trzonu kości\nłokciowej\n\n\nS62\nZłamanie na poziomie nadgarstka i\nręki\n\n\nW01.1\nUPADEK NA TYM SAMYM POZIOMIE\nWSKUTEK POTKNIECIA, POSLIZGNIECIA - INSTYTUCJA MIESZKALNA\n\n\nMAINICD10\n\n\nS52.2\nZłamanie trzonu kości\nłokciowej\n\n\n']]
ARTICLE_TO_SUMMARIZE_LIST2 = [['Informacje o pacjencie ORGUNITNAME Ort Oddział Ortopedii i Traumatologii AGE SEX CITIZENSHIP PL PLACEOFBIRTH Stoczek Łukowski POSTALCODE 05-803 FULLNAME PRUSZKÓW - GMINA MIEJSKA\n\n\n', 'Epizod Szpitalny, 2007-06-20\n20:51:00Koniec epizodu: 2007-06-22 15:11:00Event IP - Izba PrzyjęćPoczątek eventu2007-06-20 20:51:00Koniec eventu2007-06-20 20:52:00Event SOR - Szpitalny\nOddział RatunkowyPoczątek eventu2007-06-20 20:52:00Koniec eventu2007-06-20 23:01:00ICD10\n\n\nS60\nPowierzchowny uraz nadgarstka i\nręki\n\n\nMAINICD10\n\n\nS60\nPowierzchowny uraz nadgarstka i\nręki\n\n\nEvent Ort -\nOddział Ortopedii i TraumatologiiPoczątek eventu2007-06-20 23:01:00Koniec eventu2007-06-22 15:11:00ICD10\n\n\n\n\n\n\nS52\nZłamanie\nprzedramienia\n\n\nW01.1\nUPADEK NA TYM SAMYM POZIOMIE\nWSKUTEK POTKNIECIA, POSLIZGNIECIA - INSTYTUCJA MIESZKALNA\n\n\nMAINICD10\n\n\nS52\nZłamanie\nprzedramienia\n\n\n', 'Epizod Ambulatoryjny,\n2007-06-22 15:04:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-08-01 08:43:00Koniec eventu2007-08-01 09:03:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-08-08 13:55:00Koniec eventu2007-08-08 14:15:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-08-30 09:58:00Koniec eventu2007-08-30 10:18:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-09-27 10:42:00Koniec eventu2007-09-27 11:02:00Event 1005 -\nGabinet 1005 Poradni OrtopedycznejPoczątek eventu2007-11-19 15:45:00Koniec eventu2007-11-19 16:05:00Event 1007 -\nGabinet 1007 Poradni OrtopedycznejPoczątek eventu2008-06-20 09:15:00Koniec eventu2008-06-20 09:30:00', 'Epizod Szpitalny, 2007-08-21\n09:12:00Koniec epizodu: 2007-08-24 13:20:00Event IP - Izba PrzyjęćPoczątek eventu2007-08-21 09:12:00Koniec eventu2007-08-21 09:13:00ICD10\n\n\nS62\nZłamanie na poziomie nadgarstka i\nręki\n\n\nEvent Ort -\nOddział Ortopedii i TraumatologiiPoczątek eventu2007-08-21 09:13:00Koniec eventu2007-08-24 13:20:00ICD10\n\n\n\n\n\n\nS52.2\nZłamanie trzonu kości\nłokciowej\n\n\nS62\nZłamanie na poziomie nadgarstka i\nręki\n\n\nW01.1\nUPADEK NA TYM SAMYM POZIOMIE\nWSKUTEK POTKNIECIA, POSLIZGNIECIA - INSTYTUCJA MIESZKALNA\n\n\nMAINICD10\n\n\nS52.2\nZłamanie trzonu kości\nłokciowej\n\n\n']]

ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
ARTICLE_TO_SUMMARIZE1 = "My friends are cool but they eat too many carbs. My friends are cool but they eat too many carbs. Ala ma kota. jestem michal do."
ARTICLE_TO_SUMMARIZE2 = "Informacje o pacjencie ORGUNITNAME Ort Oddział Ortopedii i Traumatologii AGE SEX CITIZENSHIP PL PLACEOFBIRTH Stoczek Łukowski POSTALCODE 05-803 FULLNAME PRUSZKÓW - GMINA MIEJSKA"

if __name__ == '__main__':
    tokenizer = BartTokenizerFast.from_pretrained(
        "/home/michal/dev/abstrakt/abstrakt_inference_optimization/models/2023_04_18_polish_bart_base_chunked")
    config = AutoConfig.from_pretrained(
        "/home/michal/dev/abstrakt/abstrakt_inference_optimization/models/2023_04_18_polish_bart_base_chunked")

    # inputs = tokenizer([ARTICLE_TO_SUMMARIZE_LIST2[0][0]], max_length=500, return_tensors="pt").to('cpu')
    inputs = tokenizer([ARTICLE_TO_SUMMARIZE2], max_length=1000, return_tensors="pt").to('cpu')

    ort_sess = rt.InferenceSession(
        "/home/michal/dev/abstrakt/transformers/examples/research_projects/onnx/summarization/optimized_BART.onnx")

    ort_out = ort_sess.run(
        None,
        {
            "input_ids": inputs["input_ids"].cpu().numpy(),
            "attention_mask": inputs["attention_mask"].cpu().numpy(),
            "num_beams": np.array(1),
            "max_length": np.array(500),
            "decoder_start_token_id": np.array(config.decoder_start_token_id),
        },
    )

    decode = tokenizer.decode(ort_out[0].tolist()[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(decode)
