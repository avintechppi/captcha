# Standard imports
import re
import shutil
from time import sleep

# Third-party imports
import cv2
import numpy as np
import requests
from PIL import Image
from ultralytics import YOLO
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
#from functions import *
import speech_recognition as sr
import urllib
import subprocess
from selenium.webdriver.common.action_chains import ActionChains

import undetected_chromedriver as webdriver
from fake_useragent import UserAgent

def random_delay(mu=0.3, sigma=0.1): 
    """
    Random delay to simulate human behavior.
    :param mu: mean of normal distribution.
    :param sigma: standard deviation of normal distribution.
    """
    try:
        delay = np.random.normal(mu, sigma)
        delay = max(0.1, delay)
        sleep(delay)
    except Exception as e:
        print(f"Error in random delay: {str(e)}")

def get_target_num(driver):
    """
    Get the target number from the recaptcha title.
    """
    target_mappings = {
        "bicycle": 1,
        "bus": 5,
        "boat": 8,
        "car": 2,
        "hydrant": 10,
        "motorcycle": 3,
        "traffic": 9
    }

    target = WebDriverWait(driver, 10).until(EC.presence_of_element_located(
        (By.XPATH, '//div[@id="rc-imageselect"]//strong')))
    
    for term, value in target_mappings.items():
        if re.search(term, target.text): return value

    return 1000


def dynamic_and_selection_solver(target_num, verbose, model):
    """
    Get the answers from the recaptcha images.
    :param target_num: target number.
    :param verbose: print verbose.
    """
    # Load image and predict
    image = Image.open("0.png")
    image = np.asarray(image)
    result = model.predict(image, task="detect", verbose=verbose)

    # Get the index of the target number
    target_index = []
    count = 0
    for num in result[0].boxes.cls:
        if num == target_num: target_index.append(count)
        count += 1

    # Get the answers from the index
    answers = []
    boxes = result[0].boxes.data
    count = 0
    for i in target_index:
        target_box = boxes[i]
        p1, p2 = (int(target_box[0]), int(target_box[1])
                  ), (int(target_box[2]), int(target_box[3]))
        x1, y1 = p1
        x2, y2 = p2

        xc = (x1+x2)/2
        yc = (y1+y2)/2

        row = yc // 100 
        col = xc // 100
        answer = int(row * 3 + col + 1)
        answers.append(answer)

        count += 1

    return list(set(answers))


def get_all_captcha_img_urls(driver):
    """
    Get all the image urls from the recaptcha.
    """
    images = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, '//div[@id="rc-imageselect-target"]//img')))

    img_urls = []
    for img in images: img_urls.append(img.get_attribute("src"))

    return img_urls


def download_img(name, url):
    """
    Download the image.
    :param name: name of the image.
    :param url: url of the image.
    """

    response = requests.get(url, stream=True)
    with open(f'{name}.png', 'wb') as out_file: shutil.copyfileobj(response.raw, out_file)
    del response


def get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver):
    """
    Get all the new image urls from the recaptcha.
    :param answers: answers from the recaptcha.
    :param before_img_urls: image urls before.
    """
    images = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    img_urls = []

    # Get all the image urls
    for img in images:
        try: img_urls.append(img.get_attribute("src"))
        except:
            is_new = False
            return is_new, img_urls

    # Check if the image urls are the same as before
    index_common = []
    for answer in answers:
        if img_urls[answer-1] == before_img_urls[answer-1]: index_common.append(answer)

    # Return if the image urls are the same as before
    if len(index_common) >= 1:
        is_new = False
        return is_new, img_urls
    else:
        is_new = True
        return is_new, img_urls


def paste_new_img_on_main_img(main, new, loc):
    """
    Paste the new image on the main image.
    :param main: main image.
    :param new: new image.
    :param loc: location of the new image.
    """
    paste = np.copy(main)
    
    row = (loc - 1) // 3
    col = (loc - 1) % 3
    
    start_row, end_row = row * 100, (row + 1) * 100
    start_col, end_col = col * 100, (col + 1) * 100
    
    paste[start_row:end_row, start_col:end_col] = new
    
    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite('0.png', paste)


def get_occupied_cells(vertices):
    """
    Get the occupied cells from the vertices.
    :param vertices: vertices of the image.
    """
    occupied_cells = set()
    rows, cols = zip(*[((v-1)//4, (v-1) % 4) for v in vertices])

    for i in range(min(rows), max(rows)+1):
        for j in range(min(cols), max(cols)+1):
            occupied_cells.add(4*i + j + 1)

    return sorted(list(occupied_cells))

def square_solver(target_num, verbose, model):
    """
    Get the answers from the recaptcha images.
    :param target_num: target number.
    :param verbose: print verbose.
    """
    # Load image and predict
    image = Image.open("0.png")
    image = np.asarray(image)
    result = model.predict(image, task="detect", verbose=verbose)
    boxes = result[0].boxes.data

    target_index = []
    count = 0
    for num in result[0].boxes.cls:
        if num == target_num:
            target_index.append(count)
        count += 1

    for i in target_index:
        target_box = boxes[i]
        p1, p2 = (int(target_box[0]), int(target_box[1])
                  ), (int(target_box[2]), int(target_box[3]))
        x1, y1 = p1
        x2, y2 = p2

    answers = []
    count = 0
    for i in target_index:
        target_box = boxes[i]
        p1, p2 = (int(target_box[0]), int(target_box[1])
                  ), (int(target_box[2]), int(target_box[3]))
        x1, y1 = p1
        x4, y4 = p2
        x2 = x4
        y2 = y1
        x3 = x1
        y3 = y4
        xys = [x1, y1, x2, y2, x3, y3, x4, y4]

        four_cells = []
        for i in range(4):
            x = xys[i*2]
            y = xys[(i*2)+1]

            if x < 112.5 and y < 112.5:
                four_cells.append(1)
            if 112.5 < x < 225 and y < 112.5:
                four_cells.append(2)
            if 225 < x < 337.5 and y < 112.5:
                four_cells.append(3)
            if 337.5 < x <= 450 and y < 112.5:
                four_cells.append(4)

            if x < 112.5 and 112.5 < y < 225:
                four_cells.append(5)
            if 112.5 < x < 225 and 112.5 < y < 225:
                four_cells.append(6)
            if 225 < x < 337.5 and 112.5 < y < 225:
                four_cells.append(7)
            if 337.5 < x <= 450 and 112.5 < y < 225:
                four_cells.append(8)

            if x < 112.5 and 225 < y < 337.5:
                four_cells.append(9)
            if 112.5 < x < 225 and 225 < y < 337.5:
                four_cells.append(10)
            if 225 < x < 337.5 and 225 < y < 337.5:
                four_cells.append(11)
            if 337.5 < x <= 450 and 225 < y < 337.5:
                four_cells.append(12)

            if x < 112.5 and 337.5 < y <= 450:
                four_cells.append(13)
            if 112.5 < x < 225 and 337.5 < y <= 450:
                four_cells.append(14)
            if 225 < x < 337.5 and 337.5 < y <= 450:
                four_cells.append(15)
            if 337.5 < x <= 450 and 337.5 < y <= 450:
                four_cells.append(16)
        answer = get_occupied_cells(four_cells)
        count += 1
        for ans in answer:
            answers.append(ans)
    answers = sorted(list(answers))
    return list(set(answers))

def go_to_recaptcha_iframe1(driver):
    """
    Go to the first recaptcha iframe. (CheckBox)
    """
    driver.switch_to.default_content()
    recaptcha_iframe1 = WebDriverWait(driver=driver, timeout=20).until(
        EC.presence_of_element_located((By.XPATH, '//iframe[@title="reCAPTCHA"]')))
    driver.switch_to.frame(recaptcha_iframe1)


def go_to_recaptcha_iframe2(driver):
    """
    Go to the second recaptcha iframe. (Images)
    """
    driver.switch_to.default_content()
    recaptcha_iframe2 = WebDriverWait(driver=driver, timeout=20).until(
        EC.presence_of_element_located((By.XPATH, '//iframe[contains(@title, "challenge")]')))
    driver.switch_to.frame(recaptcha_iframe2)

def solve_recaptcha(driver, verbose):
    """
    Solve the recaptcha.
    :param driver: selenium driver.
    :param verbose: print verbose.
    """
    notes = ""
    complete = False
    try:
        go_to_recaptcha_iframe1(driver)

        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
            (By.XPATH, '//div[@class="recaptcha-checkbox-border"]'))).click()

        go_to_recaptcha_iframe2(driver)

        #Solving Image Captcha
        model = YOLO("./model.onnx", task="detect")        
        while not complete:
            for count in range(1, 2): # Object Detection up to 4 times
                print(f"Iteration {count}, complete = {complete}")
                try:
                    while True:
                        reload = WebDriverWait(driver, 10).until(
                            EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button')))
                        title_wrapper = WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.ID, 'rc-imageselect')))

                        target_num = get_target_num(driver)

                        if target_num == 1000:
                            random_delay(0.3,0.1)
                            if verbose: 
                                print("skipping")
                            reload.click()
                        elif "squares" in title_wrapper.text:
                            if verbose: 
                                print("Square captcha found....")
                            img_urls = get_all_captcha_img_urls(driver)
                            download_img(0, img_urls[0])
                            answers = square_solver(target_num, verbose, model)
                            if len(answers) >= 1 and len(answers) < 16:
                                captcha = "squares"
                                break
                            else:
                                reload.click()
                        elif "none" in title_wrapper.text:
                            if verbose: 
                                print("found a 3x3 dynamic captcha")
                            img_urls = get_all_captcha_img_urls(driver)
                            download_img(0, img_urls[0])
                            answers = dynamic_and_selection_solver(target_num, verbose, model)
                            if len(answers) > 2:
                                captcha = "dynamic"
                                break
                            else:
                                reload.click()
                        else:
                            if verbose: 
                                print("found a 3x3 one time selection captcha")
                            img_urls = get_all_captcha_img_urls(driver)
                            download_img(0, img_urls[0])
                            answers = dynamic_and_selection_solver(target_num, verbose, model)
                            if len(answers) > 2:
                                captcha = "selection"
                                break
                            else:
                                reload.click()
                        WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                            (By.XPATH, '(//div[@id="rc-imageselect-target"]//td)[1]')))

                    if captcha == "dynamic":
                        for answer in answers:
                            answerbtn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                                (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]')))
                            human_like_click(driver, answerbtn)
                            random_delay(mu=0.5, sigma=0.2)
                        while True:
                            before_img_urls = img_urls
                            while True:
                                is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                                    answers, before_img_urls, driver)
                                if is_new:
                                    break

                            new_img_index_urls = []
                            for answer in answers:
                                new_img_index_urls.append(answer-1)
                            new_img_index_urls

                            for index in new_img_index_urls: download_img(index+1, img_urls[index])
                            while True:
                                try:
                                    for answer in answers:
                                        main_img = Image.open("0.png")
                                        new_img = Image.open(f"{answer}.png")
                                        location = answer
                                        paste_new_img_on_main_img(
                                            main_img, new_img, location)
                                    break
                                except:
                                    while True:
                                        is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                                            answers, before_img_urls, driver)
                                        if is_new:
                                            break
                                    new_img_index_urls = []
                                    for answer in answers:
                                        new_img_index_urls.append(answer-1)

                                    for index in new_img_index_urls:
                                        download_img(index+1, img_urls[index])

                            answers = dynamic_and_selection_solver(target_num, verbose, model)

                            if len(answers) >= 1:
                                for answer in answers:
                                    answerbtn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                                        (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]')))
                                    human_like_click(driver, answerbtn)
                                    random_delay(mu=0.5, sigma=0.1)
                            else:
                                break
                    elif captcha == "selection" or captcha == "squares":
                        for answer in answers:
                            answerbtn = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                                (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]')))
                            human_like_click(driver, answerbtn)
                            random_delay(0.3,0.1)

                    verify = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                        (By.ID, "recaptcha-verify-button")))
                    random_delay(mu=2, sigma=0.2)
                    human_like_click(driver, verify)
                    random_delay(mu=2, sigma=0.2)
                    
                    #Check if complete, various ways depending on website
                    sleep(3)
                    go_to_recaptcha_iframe1(driver)
                    #recaptcha-checkbox goog-inline-block recaptcha-checkbox-unchecked rc-anchor-checkbox recaptcha-checkbox-checked recaptcha-checkbox-clearOutline
                    checkbox_complete = driver.find_element(By.CLASS_NAME, 'recaptcha-checkbox').get_attribute("aria-checked")
                    if checkbox_complete == "true":
                        complete = True
                        driver.switch_to.default_content()
                    else:
                        print("Challenge not complete, retrying...")

                except Exception as e:
                    print("Failed to solve image challenge:")
                    print(e)
                    print(e)
                    notes = e.args[0] if e.args else str(e)
                    break
            break
        
        if not complete:
            #do audio solve
            try:
                # click on audio challenge
                driver.switch_to.default_content()
                go_to_recaptcha_iframe2(driver)
                driver.find_element(By.ID, 'recaptcha-audio-button').click()
                random_delay(mu=2, sigma=0.2)
                
                # get the mp3 audio file
                src = driver.find_element(By.ID, 'audio-source').get_attribute("src")
                
                path_to_mp3 = "./captcha.mp3"
                path_to_wav = "captcha.wav"
            
                # download the mp3 audio file from the source
                urllib.request.urlretrieve(src, path_to_mp3)
            except Exception as e:
                print("Unable to retrieve audio challenge:")
                print(e)
                raise Exception("Unable to retrieve audio challenge.")

            # load downloaded mp3 audio file as .wav
            try:

                audio = mp3_to_wav(path_to_mp3,path_to_wav)
                
                random_delay(mu=2, sigma=0.2)
                key = speech_to_text(audio)
            except Exception as e:
                print("speech_to_text:")
                print(e)
                raise Exception("Unable to process audio.")

            # key in results and submit
            random_delay(mu=2, sigma=0.2)
            driver.find_element(By.ID,"audio-response").send_keys(key.lower())
            driver.find_element(By.ID,"audio-response").send_keys(Keys.ENTER)

            #Check if complete, various ways depending on website
            sleep(3)
            driver.switch_to.default_content()
            go_to_recaptcha_iframe1(driver)
            #recaptcha-checkbox goog-inline-block recaptcha-checkbox-unchecked rc-anchor-checkbox recaptcha-checkbox-checked recaptcha-checkbox-clearOutline
            checkbox_complete = driver.find_element(By.CLASS_NAME, 'recaptcha-checkbox').get_attribute("aria-checked")
            if checkbox_complete == "true":
                complete = True
                print("Challenge complete")
                driver.switch_to.default_content()
            else:
                print("Challenge failed to complete")
                complete = False
                notes = "Audio challenge failed to complete captcha"
        else:
            print("Captcha complete")
            complete = True
    except Exception as e:
        print("Captcha failed: ")
        print(str(e))
        notes = e.args[0] if e.args else str(e)
        complete = False
                
    return [complete,notes]

def human_like_click(driver, element):
    try:
        action = ActionChains(driver)
        action.move_to_element_with_offset(element, 0, 0).click().perform()
        print("Performing human-like click")
    except Exception as e:
        print(f"Error performing human-like click: {str(e)}")

def mp3_to_wav(path_to_mp3,path_to_wav):
    try:
        subprocess.call(["C:\\ffmpeg\\bin\\ffmpeg.exe", '-i', path_to_mp3,path_to_wav,'-y'])
        audio = sr.AudioFile(path_to_wav)
    except Exception as e:
        print("Unable to convert mp3 to wav.")
        print(str(e))
    return audio

def speech_to_text(audio):
    key = ""
    try:
        random_delay(1, 0.5)
        r = sr.Recognizer()
        with audio as source:
            sample_audio = r.record(source)
        key = r.recognize_google(sample_audio)
        print(f"Recaptcha Passcode: {key}")
    except Exception as e:
        print("speech_to_text: Speech to text error.")
        print("speech_to_text: " + str(e))
    finally:
        return key
    

url = 'https://google.com/recaptcha/api2/demo'
chrome_options = webdriver.ChromeOptions()
user_agent = UserAgent().random
user_agent = "Department of Statistics, Singapore " + str(user_agent)

chrome_options.headless = False

chrome_options.add_argument("--disable-infobars")
chrome_options.add_argument("--disable-extensions")

chrome_options.add_argument("--user-agent="+user_agent)
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument("--disable-web-security")
chrome_options.add_argument("--ignore-certificate-errors")

chrome_options.add_experimental_option("prefs", {"profile.default_content_setting_values.notifications": 2 })

driver = webdriver.Chrome(options=chrome_options)
driver.get(url)
solve_recaptcha(driver, True)
sleep(1000)