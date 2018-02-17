import os
import csv
import re
import codecs
import numpy as np
from janome.tokenizer import Tokenizer

class MakeNnc:
    def __init__(self):
        #除去する文字（空白とタブ）
        self.__sub_pat = r'[\s　\t]'
        #行分割するデリミタ（改行と。）
        self.__split_pat = r'[。\r\n]'
        #処理対象外にするする品詞（記号）
        self.__part_pat = r'記号'
        #処理状況を出力する単位（10ファイルごと）
        self.__n_break = 10

    #もとデータのパスとラベルを書いたCSV「textfile」,出力するNNCのデータセットCSV「s_outputfile」を引数にとる
    def make_data(self,textfile,s_outputfile):
        # 元データのパスと正解ラベルのデータを読み込む
        l_in = self.__get_input_csv(textfile)
        # 学習用と予測テスト用のファイルを開く
        outf_train = open(s_outputfile + "_train.csv",'w',newline='',encoding='utf8')
        outf_valid = open(s_outputfile + "_valid.csv",'w',newline='',encoding='utf8')
        csv_train_writer = csv.writer(outf_train)
        csv_train_writer.writerow(['x:image','y:label'])
        csv_valid_writer = csv.writer(outf_valid)
        csv_valid_writer.writerow(['x:image','y:label'])
        # テキストファイル１つごとに処理
        for i,l_p in enumerate(l_in):
            s_path = '.\\' + l_p[1] + '\\' + str(i) + '.csv'
            # ８：２の割合で学習データとテストデータをふりわける
            if((i % 10) > 7):
                csv_valid_writer.writerow([s_path,l_p[1]])
            else:
                csv_train_writer.writerow([s_path,l_p[1]])
            # テキストファイルを読み込む
            l_text = self.__get_input_txt(l_p[0])
            # 符号化処理を行う
            self.__make_data_csv(l_text,l_p[1],i)
            if((i > 0) and (((i % self.__n_break) == 0) or (i == len(l_in)-1))):
                print("{0}個のファイルを処理しました.".format(i))
        outf_train.close()
        outf_valid.close()
    
    # 元になるCSVファイルを読み込んでリストに変換する
    def __get_input_csv(self,csvfile):
        with open(csvfile,'r',encoding='utf8') as csvf:
           csvreader = csv.reader(csvf)
           return list(csvreader)

    # テキストファイルを読み込み、行単位のリストにして返す
    def __get_input_txt(self,textfile):
        with codecs.open(textfile,'r','utf8') as f:
            data = f.read()
            return re.split(self.__split_pat,data)
    
    # NNC用に数値データに変換したデータを1行ずつ別ファイルにして保存する
    def __make_data_csv(self,l_text,label,n_out):
        o_reg = re.compile(self.__part_pat)
        na_ar = np.array([])
        # 初期化　00からFFをひとつずつおく。位置を固定し、かつ、0エラーを回避する
        l_line = ['00','01','02','03','04','05','06','07','08','09','0A','0B','0C','0D','0E','0F']
        for fo in range(256):
            if(fo > 15):
                s = str(hex(fo)).upper()
                l_line.append(s[2:4])
        # 処理に直接関係ないが、検証時に確認しやすくなるよう元テキストも同じファイル名で出力
        os.makedirs(label,exist_ok=True)
        tout_f = open('.\\' + label + '\\' + str(n_out) + '.txt','w',newline='',encoding='utf8')
        # 形態素解析・・単語に分割するインスタンス
        o_t = Tokenizer()
        # 1行ずつ処理する
        for i,l_pt in enumerate(l_text):
            # 不要な空白などを除去
            l_p = re.sub(self.__sub_pat,"",l_pt)
            # 空行は無視する
            if l_p:
                # 元テキストも出力しておく
                tout_f.write(l_p + "\r\n")
                # 1行分の単語を分割
                o_malist = o_t.tokenize(l_p)
                for n in o_malist:            
                    # 記号以外を対象にする
                    if not (o_reg.match(n.part_of_speech)):
                        # 分割した単語を1行分リストに加えていく                    
                        for i in range(len(n.surface)):                    
                            h = str(hex(ord(n.surface[i]))).upper()
                            if(len(h) >=4):
                                l01 = h[2:4]
                                l_line.append(l01)
                            if(len(h) >=6):
                                l02 = h[4:6]
                                l_line.append(l02)
        tout_f.close()
        # コードごとにカウントする
        c_dic = {}
        t_cnt = 0
        for tok in l_line:
            c_dic.setdefault(tok,0)
            c_dic[tok] += 1
            t_cnt += 1

        # ファイルの中の出現割合に変換する
        l_output = []
        for k, v in sorted(c_dic.items()):
            rt = v / t_cnt
            l_output.append(rt)
        # 1行分ずつCSVファイルに保存する。フォルダは正解ラベル毎にわけておく
        with open('.\\' + label + '\\' + str(n_out) + '.csv','w',newline='',encoding='utf8') as df:
            cw = csv.writer(df)
            cw.writerow(l_output)
        df.close()

