import logging
import pickle
from pprint import pprint
from collections import defaultdict
from string import punctuation
from time import time
from data_collection import download_collection, connect_to_db
from bson.dbref import DBRef
from lda import LDA
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from scipy import interpolate
from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format='%(filename)s[LINE:%(lineno)d]# %(levelname)-5s [%(asctime)s]  %(message)s')

# Созданеи объектов выделения словоформ и корней
stemmer = PorterStemmer()
lemmer = WordNetLemmatizer()

# Бесполезные слова
STOP_WORDS = [stemmer.stem(w) for w in stopwords.words('english') + [
    're go', '—', 'yeah', 'okay', 'ok', 'oh', 'ye', 'bit', 'whole', 'ever', 'bit', 're', 'go', 'c', 'isn', 'per',
    're look', 're talk', 're try', 'really want', 'say go', 'say know', 'say re', 'say well', 'th']]

BAD_WORDS = word_tokenize(' '.join(list(punctuation) + stopwords.words('english') + ['s', 't', 'll', 've',
                                                                                    'd', 'm', 'c', 're']))
POSSIBLE_TAGS = {'VERB': 'v', 'NOUN': 'n', 'ADJ': 'a', 'ADV': 'r'}


def prepare_texts(return_indexes=False) -> tuple:
    """
    Создает два списка: список текстов, говых для обработка на векторизаторе, и список названиий документов, которые
    вошли в список текстов. Функция так же создает txt файл с индексами лекций, которые не вошли в список текстов,
    потому что не содержат текстов.

    :param return_indexes: Если True, то функция список индексов лекций вместо их названий
    :return: кортеж списков
    """

    # Загрузка коллекции лекций
    t0 = time()
    data = download_collection('lectures')
    texts = []
    id2text = []
    omitted_ids = []
    if return_indexes:
        # Создание списка документов
        for i in data['text'].index:
            if data.loc[i, 'text'] != '':
                texts.append(data.loc[i, 'text'])
                id2text.append(int(i))
            else:
                # Сбор документов, неимеющих тексты
                omitted_ids.append(int(i))
    else:
        for i in data['text'].index:
            if data.loc[i, 'text'] != '':
                texts.append(data.loc[i, 'text'])
                id2text.append(data.loc[i, 'title'])
            else:
                omitted_ids.append(int(i))

    logging.info('{} are texts prepared in {:.3} sec'.format(len(texts), time() - t0))
    with open('tmp/omitted_ids.txt', 'w') as file:
        for id in omitted_ids:
            file.write(str(id) + '\n')
        file.close()

    return texts, id2text


def _process_text(raw_text: str, stemming=True, lemmatize=True) -> list:
    """
    Функция обрабатывает текст, разбивая его на отдельные слова. Затем из этих слов выделюятся отедльные словоформы,
    из которых потом выделются корни. При этом из текстов удаляется вся пунктуация и бесполезные слова.

    :param raw_text: Сырой текст
    :param stemmatize: Если True - выделяет корни, False - нет
    :param lemmatize: Если True - выделяет словоформы, False - нет
    :return: Список корней слов
    """
    # Удаление пунктуации из текста
    table = str.maketrans({key: ' ' for key in list(punctuation)})
    raw_text = raw_text.translate(table)

    # Лемматизация и выделение корней
    tokens = pos_tag(word_tokenize(raw_text), 'universal')

    if lemmatize:
        result = []
        for token in tokens:
            if token[0] not in BAD_WORDS:
                if token[1] in POSSIBLE_TAGS:
                    token_prep = lemmer.lemmatize(token[0], pos=POSSIBLE_TAGS[token[1]])
                    result.append(token_prep)
                else:
                    result.append(token[0])
        tokens = result

    if stemming:
        tokens = [stemmer.stem(token) for token in tokens]

    return tokens


def get_corpus(texts: list, vectorizer, min_df=0.1, max_df=0.5):
    """
    Создает корпус текстов в формате [n_samples, n_features].
    Если такой корпус уже существует, то он загружается.

    :param texts: Список кластеризуемых текстов
    :param min_df: Минимальная частота, с которой слово должно встречаться в текстах; слова, частота которых ниже
    данной частоты отбарсываются
    :param max_df: Максимальная частота, с которой слово должно встречаться в текстах; слова, частота которых выше
    данной частоты отбарсываются Tf или Tf-idf
    :return: Корпус документов и списко слов
    """
    # Проверка типа векторизатора
    if issubclass(vectorizer, TfidfVectorizer):
        corpus_type = 'Tfidf'
    else:
        corpus_type = 'Tf'

    # Создание имен файлов, соответствующих векторизатору
    coprus_filename = 'tmp/{}_corpus.pickle'.format(corpus_type)
    features_filename = 'tmp/{}_feature_names.pickle'.format(corpus_type)

    # Попытка загрузить уже существующий корпус
    try:
        corpus = pickle.load(open(coprus_filename, 'rb'))
        feature_names = pickle.load(open(features_filename, 'rb'))
        logging.info('{} Corpus and feature names uploaded'.format(corpus_type))

    # В случае неудачной попытки корпус созадется заново
    except FileNotFoundError:
        logging.info('Corpus has not been found')
        logging.info('Creating new one')

        # Создание объекта векторизатора
        t0 = time()
        vector = vectorizer(tokenizer=_process_text, analyzer='word', stop_words=STOP_WORDS, max_df=max_df,
                            min_df=min_df, lowercase=True)
        logging.info('{0} vectorizer created in {1:.3}sec'.format(corpus_type, time() - t0))

        # Загрузака текстов в векторизатор
        t0 = time()
        corpus = vector.fit_transform(texts)
        logging.info('{0} created in {1:.3} sec'.format(corpus_type, time() - t0))

        # Получение списка слов
        feature_names = vector.get_feature_names()
        logging.info('Feature names extracted')

        # Конвертация корпуса и слов в pickle
        pickle.dump(corpus, open(coprus_filename, 'wb'))
        pickle.dump(feature_names, open(features_filename, 'wb'))

    return corpus, feature_names


def kcluster_text(corpus, n_clusters: int, id2text: list, id2word: list, n_components: int, n_top_features=10,
                  dump_to_db=False) -> tuple:
    """
    Ф-ция проводит кластеризацию текстов методом k-средних и загружает полученные кластеры в базу данных

    :param id2text: Список имен (индексов) текстов
    :param n_clusters: Число кластеров
    :param n_components: Число самых важных слов, по которым проводится кластеризация
    :param id2word: Список слов
    :param corpus: Корпус текстов
    :param n_top_features: Число выводимых слов, характеризующих кластер
    :return: Словарь кластеров, а так же
    """

    # Понижение размерности матрицы при помощи латентно-семантического анализа
    # Выделяет n самых значимых слова (n_components), отбрасывая все остальные
    if n_components:
        t0 = time()
        svd = TruncatedSVD(n_components)
        lsa = make_pipeline(svd, Normalizer(copy=False))
        corpus = lsa.fit_transform(corpus)
        logging.info("LSA applied in {:.3}".format(time() - t0))

    # Кластеризация методом К-средних
    t0 = time()
    km_model = KMeans(n_clusters=n_clusters, verbose=-1)
    cluster_labels = km_model.fit_predict(corpus)
    logging.info('K-means performed in {0:.3}'.format(time() - t0))

    # Полученные кластеры упаковываются вместе с характерными для них лекциями в словарь
    clusters = defaultdict(list)
    for text_id, label in enumerate(km_model.labels_):
        clusters[label].append(id2text[text_id])

    # Выделяет центроиды для полученых кластеров
    if n_components:
        # Если был применен SVD метод, возвращает нормальную размерность
        original_space_centroids = svd.inverse_transform(km_model.cluster_centers_)
        order_centroids = original_space_centroids.argsort()[:, ::-1]
    else:
        order_centroids = km_model.cluster_centers_.argsort()[:, ::-1]

    # Каждому кластеру приписвыает n_top_features слов
    top_features = defaultdict(list)
    for cluster_id in range(n_clusters):
        for word_id in order_centroids[cluster_id, :n_top_features]:
            top_features[cluster_id].append(id2word[word_id])

    # Если в качестве id2text данны индексы, то происходит запись кластеров в базу данных
    if dump_to_db:
        collection = connect_to_db()['k_clusters']
        collection.drop()
        logging.info("Table '{0}_{1}' created ".format('clusters', n_clusters))
        for cluster_id in dict(clusters):
            doc = {'_id': int(cluster_id), 'top_words': top_features[cluster_id],
                   'lectures': [DBRef(collection='lectures', id=lect_id) for lect_id in clusters[int(cluster_id)]]}
            logging.info("Cluster {} dumped to database".format(int(cluster_id)))
            collection.insert(doc)

    # Сведение данных в один список
    result = [{'_id': int(cluster_id), 'top_words': top_features[cluster_id], 'titles': clusters[int(cluster_id)]}
              for cluster_id in clusters]

    return result, cluster_labels


def silhouette_analysis(corpus, cluster_labels) -> None:
    """
    Построение силуэтного графика для оценки качества проведенной кластеризации

    :param corpus: Начальный корпус текстов
    :param cluster_labels: Список текстов, приписанных к кластерам
    :return: None
    """

    # Определение числа кластеров
    n_cluster = len(np.unique(cluster_labels))

    #  Установка стиля
    plt.style.use('ggplot')

    # Создание окна для построение графика
    fig = plt.figure(n_cluster, figsize=(8, 6))
    ax1 = plt.subplot(111)
    ax1.axis([0, len(cluster_labels) + (n_cluster + 1) * 25, -0.15, 1])
    logging.debug('Figure created')

    # Подсчет среднего значения силуэтного коэффициента
    silhouette_avg = silhouette_score(corpus, cluster_labels)
    print('For n_clusters = {0} the average silhouette_score is :{1}'.format(n_cluster, silhouette_avg))

    # Подсчет силуэтного коэффициента для каждого документа
    sample_silhouette_values = silhouette_samples(corpus, cluster_labels)

    # Начальный отступ и левая граница первого кластера
    x_left = 25

    # Итерация по кластерам для пострения каждого
    for i in range(n_cluster):
        # Выбор и сортировка по убыванию значений силуэтных коэффициентов для тексто пренадлежащих i-ому кластеру
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        ith_cluster_silhouette_values = ith_cluster_silhouette_values[::-1]

        # Подсчет числа текстов в кластере
        size_cluster_i = ith_cluster_silhouette_values.shape[0]

        # Расчет правой границы i-ого кластера
        x_right = x_left + size_cluster_i

        # Заполенение графика кластера одним цветом
        color = cm.spectral(float(i) / n_cluster)
        ax1.fill_between(np.arange(x_left, x_right), 0, ith_cluster_silhouette_values, facecolor=color,
                         edgecolor=color, alpha=0.7, label='{}'.format(i))

        # Расчет левой границы следующиего кластера
        x_left = x_right + 25

    plt.title('The silhouette plot for {} clusters'.format(n_cluster))
    ax1.set_xlabel("Texts ids")
    ax1.set_ylabel("The silhouette coefficient values")

    # Отметка среднего силуэтного коэффициента для всех текстов
    ax1.axhline(y=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    ax1.set_xticks([])

    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    ax1.legend(loc='center left', fancybox=True, shadow=True, bbox_to_anchor=(1, 0.5), ncol=2)

    plt.show()


def assign_topics_lda(n_topics: int, id2text, corpus, id2word, n_top_features=10, dump_to_db=True):
    """
    Возвращает вероятностное распределение документов по темам
    :param n_topics: Число возможных тем (топиков)
    :param id2text: Список имен (индексов) текстов
    :param corpus: текстов
    :param id2word: Список слов
    :param n_top_features: Число выводимых слов, характеризующих кластер
    :param dump_to_db: True - записывает топики в базу данных, False - не записывает топики в базуданных
    :return:
    """
    t0 = time()
    lda = LDA(n_topics=n_topics, n_iter=500, random_state=1)
    logging.info('LDA created in {:.3} sec'.format(time() - t0))

    t0 = time()
    doc_topic_dist = lda.fit_transform(corpus)
    logging.info('LDA model fit-transformed in {:.3} sec'.format(time() - t0))

    if dump_to_db:
        lda_topics = connect_to_db()['lda_clusters']
        lda_topics.drop()
        for topic_idx, topic_dist in enumerate(lda.topic_word_):
            doc = {'_id': int(topic_idx),
                   'terms': [id2word[i] for i in np.argsort(topic_dist)[:-n_top_features - 1:-1]]}
            lda_topics.insert(doc)
            logging.info('Topic {} dumped to database'.format(topic_idx))

    topics = {}
    for topic_idx, topic_dist in enumerate(lda.topic_word_):
        topics[topic_idx] = [id2word[i] for i in np.argsort(topic_dist)[:-n_top_features - 1:-1]]
    docs_topics = []
    for text in doc_topic_dist:
        doc = {}
        for topic_id, topic_value in enumerate(text):
            doc[str(int(topic_id))] = topic_value
        docs_topics.append(doc)
    docs_topics = pd.DataFrame(docs_topics, index=id2text)
    return topics, docs_topics, lda


def plot_lda_topics(model, plots_per_figure=5):
    n_topics = len(model.components_)
    plt.style.use('ggplot')
    for j in range(0, n_topics, plots_per_figure):
        if j == (n_topics // plots_per_figure) * plots_per_figure:
            m = n_topics
        else:
            m = j + plots_per_figure
        f, ax = plt.subplots(m - j, 1, figsize=(8, 6), sharex=True, squeeze=False)
        for i, k in enumerate(range(j, m)):
            ax[i, 0].stem(model.topic_word_[k, :], linefmt='b-', markerfmt='bo', basefmt='w-')
            ax[i, 0].set_xlim(-10, 850)
            ax[i, 0].set_ylim(0, 0.1)
            ax[i, 0].set_ylabel("Prob")
            ax[i, 0].set_title("Topic {}".format(k))
            ave_prob = sum(model.topic_word_[k, :]) / len(model.topic_word_[k, :])
            ax[i, 0].axhline(y=ave_prob, color="red", linestyle="--")
        ax[m - j - 1, 0].set_xlabel("word")
        plt.tight_layout()
    plt.show()


def estimate_alfa(docs_procent, docs_topics, plot=True):
    """
    Функция для выбора порога
    :param docs_procent: Процент документов, которые должны войти в
    :param docs_topics: Вероятностное распределение документов по темам
    :param plot:
    :return:
    """
    logging.info('Docs_topics downloaded')
    docs_alfa = pd.DataFrame()
    for alfa in np.arange(0.0, 1, 0.005):
        for i in docs_topics.index:
            if docs_topics.loc[i].max() >= alfa:
                docs_alfa.set_value(i, str(alfa), True)
            else:
                docs_alfa.set_value(i, str(alfa), False)
        logging.info('Alfa {} counted'.format(alfa))

    procent = np.array([docs_alfa[col].sum() for col in docs_alfa.columns]) / 2040 * 100
    logging.info('Procent array created')

    if plot:
        plt.style.use('ggplot')
        plt.plot(procent[::-1], docs_alfa.columns.values.astype(float)[::-1])
        plt.axvline(x=docs_procent, color="red", linestyle="--")
        plt.ylabel("Alfa")
        plt.xlabel("% Docs assigned to some topic")
        plt.xlim(0, 100)
        plt.ylim(0, 1)
        plt.show()

    f = interpolate.interp1d(procent[::-1], docs_alfa.columns.values.astype(float)[::-1])
    return round(f(docs_procent).tolist(), 2)


def transform_docs_to_alfa(docs_topics, max_alfa=1, step_alfa=0.005):
    table_T = docs_topics.T
    topics_alfa = pd.DataFrame()
    for alfa in np.arange(0, max_alfa, step_alfa):
        logging.info('{} alfa counting'.format(alfa))
        for i in table_T.index:
            logging.info('\t {} topic counting'.format(i))
            n = 0
            for j in table_T.columns:
                if table_T.loc[i, j] >= alfa:
                    n += 1
            topics_alfa.set_value(i, str(alfa), n)
    return topics_alfa


def plot_alfa(topics_alfa):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = topics_alfa.columns.values.astype(float)
    for i in topics_alfa.index:
        z = topics_alfa.loc[i].values
        ids = np.full(z.shape, float(i))
        ax.plot(x, ids, z)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(topics_alfa.index) + 1)
    ax.set_zlim(0, 2050)

    ax.set_xlabel('Alfa')
    ax.set_ylabel('Topics')
    ax.set_zlabel('Docs')

    plt.show()


def plot_alfa_dx(topics_alfa):
    plt.style.use('ggplot')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = topics_alfa.columns.values.astype(float)
    for i in topics_alfa.index:
        y = topics_alfa.loc[i].values
        f = interpolate.interp1d(x, y)
        df_dx = np.array([derivative(f, x0, dx=1e-6) for x0 in x[1:-1]])[1:]
        ids = np.full(df_dx.shape, float(i))
        ax.plot(xs=x[1:-1][1:], ys=ids, zs=df_dx)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(topics_alfa.index) + 1)

    ax.set_xlabel('Alfa')
    ax.set_ylabel('Topics')
    ax.set_zlabel('dDocs/dAlfa')

    plt.show()


def lda_cluster_text(alfa, docs_topics, dumb_to_db=True):
    """
    Формирует кластеры и записывает их в базу данных
    :param alfa: порог
    :param docs_topics: Вероятностное распределение документов по темам
    :param dumb_to_db:
    :return:
    """
    clusters = defaultdict(list)
    topics_docs = docs_topics.T
    logging.info('Docs_topics transposed')
    for i in topics_docs.index:
        for col in topics_docs.columns:
            if topics_docs.loc[i, col] >= alfa:
                clusters[int(i)].append(int(col))
    logging.info('Clusters created')

    if dumb_to_db:
        lda_topics = connect_to_db()['lda_clusters']
        logging.info('lda_topics connected')
        for topic_idx in clusters:
            lectures = [DBRef('lectures', id) for id in clusters[topic_idx]]
            lda_topics.update({"_id": topic_idx}, {"$set": {"lectures": lectures}})
            logging.info('{} topic dumped to db'.format(topic_idx))
    return clusters


def score_clusters(collection):
    """
    Считает среднeвзвешенные оценки для каждого кластера
    :param collection:
    :return:
    """
    db = connect_to_db()
    cluster_db = db[collection]
    logging.info('Database connected')
    clusters_pd = download_collection(collection)
    logging.info('Clusters downloaded')
    for topic_idx in clusters_pd.index:
        total_views = sum([db.dereference(lecture)['views'] for lecture in clusters_pd.loc[topic_idx, 'lectures']])
        logging.info('\tTotal views for {} topic counted'.format(topic_idx))

        score_result = defaultdict(float)
        for lecture in clusters_pd.loc[topic_idx, 'lectures']:
            lec_id = db.dereference(lecture)['_id']
            scores = db.dereference(lecture)['scores']
            views = db.dereference(lecture)['views']
            for key in scores:
                score_result[key] += scores[key] * views
            logging.info('\t\t Lecture {} in topic {} counted'.format(lec_id, topic_idx))
        for key in score_result:
            score_result[key] /= total_views
            logging.info("\t'{:^12}' score in  topic {} normalized".format(key, topic_idx))
        cluster_db.update({'_id': int(topic_idx)}, {'$set': {'scores': dict(score_result)}})
        logging.info('Topic {} dumped to db'.format(topic_idx))


def k_means(n_cluster, n_components):
    texts, names = prepare_texts(return_indexes=True)
    model, features = get_corpus(texts, TfidfVectorizer)
    clusters, cluster_labels = kcluster_text(names, n_cluster, n_components, features, model)
    score_clusters('k_clusters')
    pprint(clusters, compact=True)
    silhouette_analysis(model, cluster_labels)


def lda(n_topics, doc_percent, return_indexes=True):
    texts, names = prepare_texts(return_indexes=return_indexes)
    model, features = get_corpus(texts, CountVectorizer)
    topics, docs_topics, lda_model = assign_topics_lda(n_topics, names, model, features)
    alfa = estimate_alfa(doc_percent, docs_topics)
    clusters = lda_cluster_text(alfa, docs_topics)
    score_clusters('lda_clusters')
    topics_alfa = transform_docs_to_alfa(docs_topics)
    pprint(topics, compact=True)
    pprint(clusters, compact=True)
    plot_lda_topics(lda_model, plots_per_figure=5)
    plot_alfa(topics_alfa)
    plot_alfa_dx(topics_alfa)


if __name__ == "__main__":
    texts, id2text = prepare_texts(return_indexes=True)
    corpus, features = get_corpus(texts, CountVectorizer)
