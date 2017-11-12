import codecs, sys

def sentiment_analysis(data_file, sent_file):
    sentc = 0
    kaggc = 0
    total = 0
    fs = codecs.open(sent_file, 'a+', 'utf-8')
    
    for line in open(data_file, encoding='utf-8'):
        flag = ",Sentiment140,"
        http = "http"
        www = "www."
        com = ".com"
        #print(line)
        total += 1
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
            if len(toks) > 4 and len(toks) < 15:
                strline = tokens[1] + "\t" + strline + "\n"
                print(strline)
                fs.write(strline)
            sentc += 1
        else:
            #print(line)
            kaggc += 1
            
    print(sentc, kaggc, total)


def main():
    data_file = "/Users/hjp/Downloads/Sentiment Analysis Dataset.csv"
    sent_file = "/Users/hjp/Downloads/sent.txt"
    sentiment_analysis(data_file, sent_file)


if __name__ == "__main__":
    main()
    