import os.path
from pdf2image import convert_from_path
import aiohttp
import asyncio
import argparse
import time
import requests
from os.path import basename


async def aiohttp1(pdf_files, endpoint):
    async with aiohttp.ClientSession() as session:
        coroutines = []
        for pdf in pdf_files:
            print(f"processing of {pdf} started")
            file = {"file": open(pdf, 'rb')}
            coroutines.append(session.post(endpoint, data=file))
        responses = await asyncio.gather(*coroutines)
        for r, pdf in zip(responses, pdf_files):
            content = await r.content.read()
            filename = os.path.basename(pdf).strip('.pdf')
            with open(f'../../images_temp/async_{filename}.csv', 'wb') as f:
                f.write(content)


def sync_calls(pdf_files, endpoint):
    for pdf in pdf_files:
        filename = os.path.basename(pdf).strip('.pdf')
        print(f"processing of {pdf} started")
        file = {"file": (filename, open(pdf, 'rb'), "application/pdf")}
        resp = requests.post(endpoint, files=file)
        if resp.status_code != 200:
            print("request failed!")
        else:
            content = resp.content
            with open(f'../../images_temp/sync_{filename}.csv', 'wb') as f:
                f.write(content)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default="http://localhost:8000/predict/?dpi=256&conf=0.5&stride=0.75")
    parser.add_argument("--gen_images", help="generate the images from the pdfs",
                    action="store_true")
    args = parser.parse_args()

    files = ['../../quick_start_data/pdf/Emden76.pdf',
             '../../quick_start_data/pdf/AIF_1999_375_404.pdf',
             '../../quick_start_data/pdf/ASENS_1997_367_384.pdf']

    os.makedirs(f'../../images_temp/', exist_ok=True)
    if args.gen_images:
        for pdf_file in files:
            pdf_file_base = basename(pdf_file).strip(".pdf")
            pdf_images = convert_from_path(pdf_file, dpi=256)
            os.makedirs(f'../../images_temp/{pdf_file_base}', exist_ok=True)
            for i, image in enumerate(pdf_images):
                image.save(f'../../images_temp/{pdf_file_base}/{i}.png', fmt='png')

    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(aiohttp1(files, args.endpoint))
    end = time.time()
    print(f"time taken to process pdfs async {str(end - start)}\n")

    start = time.time()
    sync_calls(files, args.endpoint)
    end = time.time()
    print(f"time taken to process pdfs sync {str(end - start)}")
