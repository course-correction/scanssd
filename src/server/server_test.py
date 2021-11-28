import os.path

import aiohttp
import asyncio
import argparse
import time
import requests


async def aiohttp1(pdf_files, endpoint):
    async with aiohttp.ClientSession() as session:
        coroutines = []
        for pdf in pdf_files:
            file = {"pdf": open(pdf, 'rb')}
            coroutines.append(session.post(endpoint, data=file))
        responses = await asyncio.gather(*coroutines)
        print(responses)


def sync_calls(pdf_files, endpoint):

    for pdf in pdf_files:
        filename = os.path.split(pdf)[0]
        file = {"pdf": (filename, open(pdf, 'rb'), "application/pdf")}
        resp = requests.post(endpoint, files=file)
        if resp.status_code != 200:
            print("request failed!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--endpoint', default="http://localhost:8000/predict/256")
    args = parser.parse_args()

    files = ['../../quickstart/pdf/Emden76.pdf',
             '../../quickstart/pdf/AIF_1999_375_404.pdf',
             '../../quickstart/pdf/ASENS_1997_367_384.pdf']
    start = time.time()
    asyncio.run(aiohttp1(files, args.endpoint))
    end = time.time()
    print(f"time taken to process pdfs async {str(end - start)}")

    start = time.time()
    sync_calls(files, args.endpoint)
    end = time.time()
    print(f"time taken to process pdfs sync {str(end - start)}")