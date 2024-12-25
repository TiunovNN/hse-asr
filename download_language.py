import asyncio
import zlib

import aiohttp
import tqdm


async def download(url, session):
    print(f'Downloading {url}')
    async with session.get(url) as response:
        response.raise_for_status()
        size = response.headers.get('content-length')
        if size:
            size = int(size)
        with tqdm.tqdm(total=size, unit='bytes', unit_scale=True) as pbar:
            async for chunk in response.content.iter_any():
                pbar.update(len(chunk))
                yield chunk


async def main():
    session = aiohttp.ClientSession()
    async with session:
        print(f'')
        with open('librispeech-vocab.txt', 'wb', buffering=2**20) as f:
            async for chunk in download('https://storage.yandexcloud.net/asr-tiunovnn/librispeech-vocab.txt', session):
                f.write(chunk)

        decompressor = zlib.decompressobj(zlib.MAX_WBITS + 32)
        with open('lowercase_3-gram.pruned.1e-7.arpa', 'wb', buffering=2**20) as f:
            async for chunk in download('https://storage.yandexcloud.net/asr-tiunovnn/lowercase_3-gram.pruned.1e-7.arpa.gz', session):
                    f.write(decompressor.decompress(chunk))

if __name__ == '__main__':
    asyncio.run(main())
