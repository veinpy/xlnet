import sentencepiece as spm

"""
special symbols:
    [CLS]:  is inserted at the beginning of the first sentence
    [SEP]:  is inserted at the end of each sentence.
    [EOP]:  End of Paragraph
    [EOD]:  End of Document
"""

spm.SentencePieceTrainer.Train("""--input=spm_test.txt \
        --model_prefix=m_test_v2 \
        --vocab_size=100000 \
        --control_symbols=<cls>,<sep>,<pad>,<mask>,<eod> \
        --model_type=unigram \
        --use_all_vocab true
        --user_defined_symbols=<eop>,.,(,),\",-,–,£,€""")


sp = spm.SentencePieceProcessor()
sp.Load("m_test_v2.model")

sp.encode_as_ids("""我不能确定对方是不是喜欢我，我却想分分秒秒跟他在一起，有谁能告诉我如何能想他少一点
一定要告诉他你很喜欢他 很爱他!!  虽然不知道你和他现在的关系是什么！但如果真的觉得很喜欢就向他表白啊！！起码你努力过了！  女生主动多少占一点优势的！！呵呵  只愿曾经拥有！  到以后就算感情没现在这么强烈了也不会觉得遗憾啊~！  与其每天那么痛苦的想他 恋他 还不如直接告诉他 ！  不要怕回破坏你们现有的感情！因为如果不告诉他  你可能回后悔一辈子！！  

""")



