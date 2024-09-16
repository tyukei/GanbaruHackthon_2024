import numpy as np
import networkx as nx
import argparse
import asyncio
import gzip
import geopy
import os
import openpyxl
import requests
import arweave
from PIL import Image
import scipy.linalg as la  # 線形代数
import sympy as sp  # 微分方程式や数学的処理
from scipy.fftpack import fft, ifft  # フーリエ変換

# ASCIIに調整する関数
def adjust_to_ascii(value, target_value):
    adjusted_value = int(np.round(value))
    # 誤差が生じないよう、範囲外の値は目標値に戻す
    if adjusted_value < 0 or adjusted_value > 255:
        os.write(1, f"Adjusted value {adjusted_value} out of range, setting to target {target_value}\n".encode('utf-8'))
        return target_value
    return adjusted_value

# NumPyで行列計算を使って最初の文字のASCII値に近づける
def numpy_step():
    A = np.array([[2, 3], [1, 4]])
    B = np.array([[1], [2]])
    result = np.dot(A, B)  # 行列計算
    target_value = ord('H')  # 'H'のASCII
    calculated_value = adjust_to_ascii(result[0][0], target_value)
    os.write(1, f"NumPy Step Calculated Value: {calculated_value}\n".encode('utf-8'))  # 出力
    return calculated_value

# NetworkXでランダムウォークを使って次の文字のASCII値に近づける
def networkx_step(previous_value):
    G = nx.path_graph(10)
    random_walk = nx.single_source_shortest_path_length(G, 0)  # ランダムウォーク的な処理
    new_value = previous_value + sum(random_walk.values()) / len(random_walk)  # 次のステップの計算
    target_value = ord('e')
    calculated_value = adjust_to_ascii(new_value, target_value)
    os.write(1, f"NetworkX Step Calculated Value: {calculated_value}\n".encode('utf-8'))
    return calculated_value

# argparseで収束を意識した数列生成によって次の値を計算
def argparse_step(previous_value):
    n = 10
    sequence = [previous_value / (i + 1) for i in range(1, n+1)]  # 収束数列
    next_value = sum(sequence)
    target_value = ord('l')
    calculated_value = adjust_to_ascii(next_value, target_value)
    os.write(1, f"Argparse Step Calculated Value: {calculated_value}\n".encode('utf-8'))
    return calculated_value

# asyncioでフーリエ変換を使って次の値を計算
async def async_step(previous_value):
    x = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * 50 * x) + np.sin(2 * np.pi * 80 * x)  # サンプル信号
    yf = fft(y)  # フーリエ変換
    transformed_value = np.abs(yf[1])
    target_value = ord('l')
    calculated_value = adjust_to_ascii(transformed_value, target_value)
    os.write(1, f"Asyncio Step Calculated Value: {calculated_value}\n".encode('utf-8'))
    await asyncio.sleep(0.1)
    return calculated_value

# gzipで微分方程式を解いて次の文字のASCII値に近づける
def gzip_step(previous_value):
    x = sp.Symbol('x')
    eq = sp.Eq(sp.Derivative(x**2 + previous_value*x, x), 0)  # 微分方程式
    solution = sp.solve(eq)
    target_value = ord('o')
    if solution:
        next_value = solution[0]
    else:
        next_value = previous_value
    calculated_value = adjust_to_ascii(next_value, target_value)
    os.write(1, f"Gzip Step Calculated Value: {calculated_value}\n".encode('utf-8'))
    return calculated_value

# geopyで極座標変換を使って次の文字（空白）のASCII値に近づける
def geopy_step(previous_value):
    r = np.sqrt(previous_value**2 + 1)
    theta = np.arctan2(1, previous_value)  # 極座標変換
    new_value = r * np.cos(theta)
    target_value = ord(' ')
    calculated_value = adjust_to_ascii(new_value, target_value)
    os.write(1, f"Geopy Step Calculated Value: {calculated_value}\n".encode('utf-8'))
    return calculated_value

# openpyxlで行列の固有値を求めて次の文字のASCII値に近づける
def openpyxl_step(previous_value):
    A = np.array([[previous_value, 2], [3, 4]])
    eigenvalues, _ = la.eig(A)
    next_value = eigenvalues[0].real  # 固有値を使う
    target_value = ord('W')
    calculated_value = adjust_to_ascii(next_value, target_value)
    os.write(1, f"Openpyxl Step Calculated Value: {calculated_value}\n".encode('utf-8'))
    return calculated_value

# ステガノグラフィ関連の処理

# メッセージをバイナリ形式に変換
def message_to_binary(message):
    return ''.join(format(ord(char), '08b') for char in message)

# バイナリ形式のメッセージを画像のピクセルに埋め込む
def hide_message_in_image(image, message):
    binary_message = message_to_binary(message)
    binary_message += '1111111111111110'  # メッセージ終了マーク
    
    img_array = np.array(image)
    flat_pixels = img_array.flatten()

    for i in range(len(binary_message)):
        flat_pixels[i] = (flat_pixels[i] & 0xFE) | int(binary_message[i])

    img_array = flat_pixels.reshape(img_array.shape)
    return Image.fromarray(img_array)

# メッセージを画像から抽出する
def extract_message_from_image(image):
    img_array = np.array(image)
    flat_pixels = img_array.flatten()

    binary_message = ''.join(str(pixel & 1) for pixel in flat_pixels)
    end_marker = '1111111111111110'  # メッセージ終了マーク
    end_index = binary_message.find(end_marker)

    if end_index == -1:
        end_index = len(binary_message)

    binary_message = binary_message[:end_index]
    return ''.join(chr(int(binary_message[i:i+8], 2)) for i in range(0, len(binary_message), 8))

# 画像から隠されたメッセージを抽出する
def read_message_from_image(image_path):
    # 画像を読み込む
    image = Image.open(image_path)
    
    # メッセージを抽出
    extracted_message = extract_message_from_image(image)
    
    # 抽出されたメッセージを返す
    return extracted_message

# Arweaveに画像をアップロード
def upload_to_arweave(wallet_path, image_path):
    wallet = arweave.Wallet(wallet_path)
    with open(image_path, 'rb') as file:
        image_data = file.read()

    transaction = arweave.Transaction(wallet, data=image_data)
    transaction.add_tag('Content-Type', 'image/png')
    transaction.sign()
    transaction.send()

    return f"https://arweave.net/{transaction.id}"

# Arweaveから画像をダウンロード
def download_image_from_arweave(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        os.write(1, f"Image downloaded from {url} and saved as {save_path}\n".encode('utf-8'))
    else:
        raise Exception(f"Failed to download image from {url}, status code: {response.status_code}")

# 最終的にHello Worldを表示する関数
def os_write_step():
    # 各ライブラリを使って高度な数式処理を行い、最終的にHello Worldを生成
    message = ''.join(chr(c) for c in [
        numpy_step(), networkx_step(numpy_step()), argparse_step(networkx_step(numpy_step())),
        asyncio.run(async_step(argparse_step(networkx_step(numpy_step())))),
        gzip_step(asyncio.run(async_step(argparse_step(networkx_step(numpy_step()))))),
        geopy_step(gzip_step(asyncio.run(async_step(argparse_step(networkx_step(numpy_step())))))),
        openpyxl_step(geopy_step(gzip_step(asyncio.run(async_step(argparse_step(networkx_step(numpy_step()))))))),
        ord('o'), ord('r'), ord('l'), ord('d')
    ]) + '\n'
    os.write(1, f"Final Generated Message: {message}".encode('utf-8'))
    os.write(1, message.encode('utf-8'))  # バイナリ形式で出力

# メインの実行フロー
def main():
    # ステガノグラフィでメッセージを隠す
    input_image_path = 'eisa.png'  # 入力画像パス
    output_image_path = 'steganography_image_with_message.png'
    wallet_path = 'wallet.json'  # Arweaveウォレットのパス

    # ステップを経由して最終的なメッセージを生成
    os_write_step()

    # 画像を読み込み、「Hello World」を隠す
    image = Image.open(input_image_path).convert('RGB')
    message = "Hello World"
    image_with_message = hide_message_in_image(image, message)
    image_with_message.save(output_image_path)

    os.write(1, f"Message hidden in image and saved as {output_image_path}\n".encode('utf-8'))

    # Arweaveにアップロード
    arweave_url = upload_to_arweave(wallet_path, output_image_path)
    os.write(1, f"Image uploaded to Arweave: {arweave_url}\n".encode('utf-8'))

    # ダウンロードしてメッセージを抽出
    download_image_path = 'downloaded_steganography_image.png'
    download_image_from_arweave(arweave_url, download_image_path)

    # ダウンロードした画像からメッセージを抽出
    extracted_message = read_message_from_image(download_image_path)
    os.write(1, f"Extracted Message: {extracted_message}\n".encode('utf-8'))

if __name__ == "__main__":
    main()