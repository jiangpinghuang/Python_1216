import os
import nltk
import os.path
import codecs, sys


def sait_data(data_file, sent_file):
    print("Generating sait data, please waiting...")
    sentc, kaggc, total = 0, 0, 0
    fs = codecs.open(sent_file, 'a+', 'utf-8')
    
    for line in open(data_file, encoding='utf-8'):
        flag, http, www, com = ",Sentiment140,", "http", "www.", ".com"
                
        if flag in line and http not in line and www not in line and com not in line:        
            sents = line.split(flag)
            tokens = sents[0].split(",")
            strline = sents[1].replace("\/", " \ / ")
            strline = strline.replace("?", " ? ")
            strline = strline.replace("!", " ! ")
            strline = strline.replace("..", " ")
            strline = strline.replace("--", " ")
            strline = strline.replace("? ?", "?")
            strline = strline.replace(" - ", " ")
            strline = strline.replace("#", " # ")
            strline = strline.replace("$", " $ ")
            strline = strline.replace("&", " & ")
            strline = strline.replace("*", " * ")
            strline = strline.replace("@", " @ ")
            strline = strline.replace("\*", " \ * ")
            strline = strline.replace("******", " * ")
            strline = strline.replace("# ", " ")
            strline = strline.replace("@ ", " ")
            strline = strline.replace("- ", " ")
            strline = strline.replace("! ", " ")
            strline = strline.replace(". ", " ")
            strline = strline.replace(": ", " ")
            strline = strline.replace(" \"", " ")
            strline = strline.replace("* ", " ")
            strline = strline.replace("& ", " ")
            strline = strline.replace("\" ", " ")
            strline = strline.replace("   ", " ")
            strline = strline.replace("   ", " ")
            strline = strline.replace("  ", " ")
            strline = strline.strip()
            toks = strline.split()
            if len(toks) > 4 and len(toks) < 20:
                strline = tokens[1] + "\t" + strline + "\n"
                fs.write(strline)
            sentc += 1
        else:
            kaggc += 1
        total += 1  
         
    print(sentc, kaggc, total)
    print("Generating sait data finished!")
    

def imdb_data(imdb_dir, imdb_file):
    print("Generating imdb data, please waiting...")
    fs = codecs.open(imdb_file, 'a+', 'utf-8')
    sent_count = 0
    
    for parent, dirnames, filenames in os.walk(imdb_dir):            
        for filename in filenames:
            if "._" not in filename:
                with open(os.path.join(parent, filename), encoding='utf-8') as f:
                    label = filename.split('.')[0].split('_')[1]
                    for line in f:  
                        strline = line.replace("<br /><br />", " ")   
                        strline = strline.replace("- &", " ") 
                        strline = strline.replace(" - ", " ")
                        strline = strline.replace("  ", " ")           
                        tokens = strline.split()                        
                        if len(tokens) < 85 and len(tokens) > 10:  
                            sents = ""
                            words = nltk.word_tokenize(strline)                              
                            for word in words:
                                if len(sents) < 1:
                                    sents = word
                                else:
                                    sents = sents + " " + word
                            sent_count += 1
                            fs.write(str(int(label) - 1) + "\t" + sents + "\n")  
                              
    fs.close()
    print(sent_count)   
    print("Generating imdb data finished!") 


def main():
    print("The program is running, please waiting...")
    sait_data_file = "/Volumes/SanDisk/Workshop/Corpus/sait/src/sait.csv"
    sait_sent_file = "/Volumes/SanDisk/Workshop/Corpus/sait/data/sait.txt"
    sait_data(sait_data_file, sait_sent_file)
    
    imdb_train_dir = "/Volumes/SanDisk/Workshop/Corpus/imdb/train"
    imdb_test_dir = "/Volumes/SanDisk/Workshop/Corpus/imdb/test"
    imdb_train_file = "/Volumes/SanDisk/Workshop/Corpus/imdb/data/train.txt"
    imdb_test_file = "/Volumes/SanDisk/Workshop/Corpus/imdb/data/test.txt"
    imdb_data(imdb_train_dir, imdb_train_file)
    imdb_data(imdb_test_dir, imdb_test_file)
    print("The program has finished data generation, check the data please!")


if __name__ == "__main__":
    main()
    
