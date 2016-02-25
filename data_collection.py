from requests import get
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re
import json
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(filename)s[LINE:%(lineno)d]# %(levelname)-5s [%(asctime)s]  %(message)s')


def get_page(index: int) -> BeautifulSoup:
    """
    Функция, подставляя индекс в ссылку с помошщью фунции get() получает
    объект страницы. С пощощью raw_page.text по ф-ция получает html строку
    страницы и парсит ее, превращяя в объект страницы.

    :param index: Индекс страницы/лекции в базе данных ted.com
    :return: Объект html страницы BeautifulSoup c номером i
    """
    # скачивание html страницы
    url = 'http://ted.com/talks/{0}'.format(index)
    raw_page = get(url)
    page = BeautifulSoup(raw_page.text, 'html.parser')

    return page


def get_info(page) -> tuple:
    """
    Ф-ция получает страницы файл json в одном из скриптов и получает из него
    всю возможную инфорацию.

    Оценки считаются по процентамм, т.е. сколько процентов людей охарактеризовали тем или иным словом

    :param page: Объект html страниы BeautifulSoup
    :return: Кортеж с закогловком лекции, тегами и оценками
    """

    # Вырезание текствого json и загрузка его в файл
    lines = page.find('div', {'class': 'main talks-main'}).find_all('script')

    a = str(lines[-1]).strip('<script>').lstrip('q("talkPage.init",').rstrip(')</')
    all_info = json.loads(a)

    # загрузка оценок в процентах
    ratings = {}
    totals = sum([item['count'] for item in all_info['ratings']])
    for item in all_info['ratings']:
        ratings[item['name']] = item['count'] / totals

    # загрузка тегов из json
    targeting = all_info['talks'][0]['targeting']['tag'].split(',')

    # загрузка названия из json
    title = all_info['talks'][0]['title']

    return title, targeting, ratings


def get_views(page) -> int:
    """
    Функция находит и возвращает число просмотров лекции
    :param page: Обект html страниы BeautifulSoup
    :return: Число просмотров лекции
    """

    # число просмотров
    views = int(str(page.find('span', {'class': 'talk-sharing__value'}).string
                    ).strip('\n').replace(',', ''))

    return views


def get_text(index: int, lang='en') -> str:
    """
    Функция скачивает и обрабатыввает полученный текст лекции

    :param index: Индекс страницы/лекции в базе данных ted.com
    :param lang: Язык субтитров
    :return: Текст cубтитров
    """

    # Создает ссылку на субтитры
    raw_link = 'http://www.ted.com/talks/subtitles/id/{0}/lang/{1}/format/srt'
    link = raw_link.format(index, lang)

    # Загружает сырой текст субтитров и проводит первыичную обрабтку текста
    raw_text = get(link).text
    lines = re.findall("\D+\\n", raw_text)
    text_lines = []
    for line in lines:
        line = line.strip('\n')
        line = line.rstrip('\n')
        line = line.replace('\n', ' ')
        line = line.replace('`', '')
        line = line.replace('(Applause)', '')
        line = line.replace('(Laughter)', '')
        line = line.replace('(Mock sob)', '')
        text_lines.append(line)

    text = ' '.join(text_lines)

    return text


def connect_to_db(host='localhost', port=27017, db='ted_data'):
    """
    Ф-ция через клиент поключатся к базе данных и, если такой не существует,
    создаёт новую

    :param host: Хост подключения базы данных, изначально локальной
    :param port: Порт поклюяния базы данных
    :param db: Имя базы данных
    :return: Базы данных
    """
    client = MongoClient(host, port)
    db = client[db]
    return db


def download_collection(collection) -> pd.DataFrame:
    """
    Функция подключатеся к базе данных, скачивает определенную коллеккцию
    и предстваляет ее ввиде объекта DataFrame (pandas)

    :param collection: Имя коллекции в базе данных db
    :return: Объект pandas DataFrame содержащий коллекцию

    """

    # Подключение к безе данных
    db = connect_to_db()

    # Загрузка в DataFrame в зависимости от структуры коллекции
    data = list(db[collection].find().sort([('_id', 1)]))
    df = pd.DataFrame(data)
    return df.set_index('_id')


def download_ted_lectures(i=1, collection='lectures') -> None:
    """
    Функция заполняет базу данных лекция с ted.com

    :param collection:
    :param i: Индекс первой лекции
    :return: None
    """

    # Подключает колекцию, куда буду загружаться файлы
    lectures = connect_to_db()[collection]
    error = 0
    while error <= 10:
        try:
            page = get_page(i)
            logging.info("Page {0} downloaded".format(i))
            info = get_info(page)
            result = {'title': info[0], 'tags': info[1], 'scores': info[2], 'views': get_views(page),
                      'text': get_text(i), '_id': i}
            lectures.insert_one(result)
            logging.info('Lecture {0} dumped to database'.format(i))
            error = 0
            i += 1
        except AttributeError:
            logging.info('Lecture {0} does not exist'.format(i))
            i += 1
            error += 1
    file = open('last_lecture_id.txt', 'w')
    file.write(str(i))
    file.close()


def download_ted_lecture(i, collection='lectures') -> None:
    """
    Действует аналогично upload_ted_lectures() только загружает одну лекцию
    :return: Индекс лекции
    """
    lectures = connect_to_db()[collection]
    try:
        page = get_page(i)
        logging.info("Page {0} downloaded".format(i))
        info = get_info(page)
        result = {'title': info[0], 'tags': info[1], 'scores': info[2], 'views': get_views(page), 'text': get_text(i),
                  '_id': i}
        lectures.insert_one(result)
        logging.info('Lecture {0} dumped to database'.format(i))
    except AttributeError:
        logging.info('Lecture {0} does not exist'.format(i))


if __name__ == "__main__":
    i = int(input())
    download_ted_lecture(i=i)
