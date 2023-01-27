import nltk
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import words
import numpy as np
import matplotlib.pyplot as plt
import time
from wordcloud import WordCloud


def display_tokens_info(tokens):
    ''' Fonction permettant d'afficher le nombre de tokens et le nombre de tokens uniques'''
    
    print(f"nb tokens {len(tokens)}, nb tokens unique {len(set(tokens))}")
    print(tokens[:10]);
    
    
def tokenizer_categ(sentence, position = 0) :
    '''Tokenizer l'arbre de catégorie et extraire une catégorie selon sa position dans l'arbre'''
    
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    categ = sentence.split(">>")[position]
    word_tokens = tokenizer.tokenize(categ.lower())
    transf_desc_text = ' '.join(word_tokens)

    return transf_desc_text;


def all_categories(df, col, deep_of_tree, df_to):
    '''créer une ou plusieurs colonnes avec la catégorie provenant de l'arbre des catégories'''
  
    for i in range(deep_of_tree):
        df_to["categ_"+str(i)] = df[col].apply(lambda x : tokenizer_categ(x,i))
    return df_to;

def process_text(doc,
                 rejoin=False,
                 lemm_or_stemm=None,
                 list_rare_words=None,
                 tag_to_keep=None,
                 min_len_word=3,
                 force_is_alpha=True,
                 engl_words=False,
                 keep_brands=False,
                 extra_words=None,
                 no_duplicate=False):
    
    '''Fonction de nettoyage pour les descriptions
    
    Paramètres :
    doc : string à nettoyer
    rejoin : bool : renvoi liste ou string
    lemm_or_stemm : None, lemm, stemm : lemmatizer ou stemmatizer
    list_rare_words : None, list : liste de mot rare à ne pas conserver
    tag_to_keep : None, list of tags : POS à garder ex : ["NOUN","VERB"]
    min_len_word : int : longueur de mot minimale
    force_is_alpha : bool : garder les numeriques ou non
    engl_words : bool : garder uniquement les mots anglais
    keep_brands : bool : forcer de garder les marques 
    extra_words : list : liste de mots à exclure
    no_duplicate : bool : ne garder aucun duplicatas dans la description'''
    
    
    
    with open("brands", "rb") as fp:   # Unpickling
        brand = pickle.load(fp)
    if not extra_words:
        extra_words = []
    else:    
        with open(extra_words, "rb") as fp:   # Unpickling
            extra_words = pickle.load(fp)
    stop_words = set(stopwords.words("english"))
        
    if lemm_or_stemm == "lem":
        trans = WordNetLemmatizer()
        brand = [trans.lemmatize(w) for w in brand]
    
    if not list_rare_words:
        list_rare_words = []
    else:
        with open(list_rare_words, "rb") as fp:   # Unpickling
            list_rare_words = pickle.load(fp)
        if keep_brands:
            list_rare_words = [x for x in list_rare_words if x not in brand]

    # lower
    doc = doc.lower().strip()

    # tokenize
    tokenizer = RegexpTokenizer(r"\w+")
    raw_tokens_list = tokenizer.tokenize(doc)

    # stop words
    stop_words = set(stopwords.words("english"))
    cleaned_tokens_list = [w for w in raw_tokens_list if w not in stop_words]

    # no more len words
    more_than_N = [w for w in cleaned_tokens_list if len(w) >= min_len_word]

    # only alpha chars
    if force_is_alpha:
        alpha_tokens = [w for w in more_than_N if w.isalpha()]
    else:
        alpha_tokens = more_than_N

    # tag_pos
    if tag_to_keep:
        tags = nltk.pos_tag(alpha_tokens, tagset="universal")
        tag_to_keep_tokens = []
        for i in tag_to_keep:
            tag_to_keep_tokens.extend([w for w, pos in tags if (pos == i)])
    else:
        tag_to_keep_tokens = alpha_tokens

    # stem or lem
    if lemm_or_stemm == "lem":
        trans = WordNetLemmatizer()
        trans_text = [trans.lemmatize(w) for w in tag_to_keep_tokens]
    elif lemm_or_stemm == "stem":
        trans = PorterStemmer()
        trans_text = [trans.stem(w) for w in tag_to_keep_tokens]
    else:
        trans_text = tag_to_keep_tokens
    # no rare tokens
    non_rare_tokens = [w for w in trans_text if w not in list_rare_words]

    # In English
    if engl_words:
        eng_words = [w.lower() for w in words.words()]
        if keep_brands:
            eng_words.extend(brand)
        else:
            eng_words = eng_words
        engl_text = [w for w in non_rare_tokens if w in eng_words]
    else:
        engl_text = non_rare_tokens

    # drop extra words tokens
    if extra_words:
        no_extra_words = [w for w in engl_text if w not in extra_words]
    else:
        no_extra_words = engl_text

    # remove duplicates
    if no_duplicate:
        final = list(set(no_extra_words))
    else:
        final = no_extra_words

    # Manage return type
    if rejoin:
        return " ".join(final)

    return final

from sklearn import manifold, decomposition
from sklearn import cluster, metrics


# Calcul Tsne, détermination des clusters et calcul ARI entre vrais catégorie et n° de clusters
def ARI_fct(features, l_cat , y_cat_num) :
    time1 = time.time()
    num_labels=len(l_cat)
    tsne = manifold.TSNE(n_components=2, perplexity=30, n_iter=2000, 
                                 init='random', learning_rate=200, random_state=42)
    X_tsne = tsne.fit_transform(features)
    
    # Détermination des clusters à partir des données après Tsne 
    cls = cluster.KMeans(n_clusters=num_labels, n_init=100, random_state=42)
    cls.fit(X_tsne)
    ARI = np.round(metrics.adjusted_rand_score(y_cat_num, cls.labels_),4)
    time2 = np.round(time.time() - time1,0)
    print("ARI : ", ARI, "time : ", time2)
    
    return ARI, X_tsne, cls.labels_


# visualisation du Tsne selon les vraies catégories et selon les clusters
def TSNE_visu_fct(X_tsne, y_cat_num, l_cat, labels, ARI) :
    fig = plt.figure(figsize=(15,6))
    
    ax = fig.add_subplot(121)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=y_cat_num, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=l_cat, loc="best", title="Categorie")
    plt.title('Représentation des produits par catégories réelles')
    
    ax = fig.add_subplot(122)
    scatter = ax.scatter(X_tsne[:,0],X_tsne[:,1], c=labels, cmap='Set1')
    ax.legend(handles=scatter.legend_elements()[0], labels=set(labels), loc="best", title="Clusters")
    plt.title('Représentation des produits par clusters')
    
    plt.show()
    print("ARI : ", ARI)
    
    
def wordcloud_generator(token):
    wordcloud = WordCloud(background_color = 'white',
                     stopwords = [],
                    collocations = False,
                     max_words = 50).generate(" ".join(token))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()