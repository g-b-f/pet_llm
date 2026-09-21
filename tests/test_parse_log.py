"""Tests for :mod:`parse_log` and the :mod:`lib.types.log` models it builds."""

from datetime import datetime as dt
from pathlib import Path

import pytest

import parse_log
from lib.types.config import ParamsConfig
from lib.types.log import (
    EventType,
    LogTrial,
    MalformedJSONEvent,
    MemoryClearEvent,
    MemoryClearReason,
    OptimisationLog,
    SeedRun,
    ThoughtEvent,
)

# A representative slice of the optimisation log: a header, one trial with
# params (two seeds, one with a loss) and one bare-seed trial (no params).
SAMPLE_LOG = """\
2026-09-05 19:40:42 INFO - starting optimisation
2026-09-05 19:40:42 INFO - RUNTIME=200, N_TRIALS=20, N_SEEDS=3
2026-09-05 19:40:42 INFO - eta: 3 hours from now
2026-09-05 19:40:44 INFO - seed=0, context_size=2048, temperature=1.45, frequency_penalty=2.2, presence_penalty=0.7, repeat_penalty=2.4, min_p=0.05, seed=0
2026-09-05 19:40:49 DEBUG - system prompt hash: 4a58e968926ee212e2dea3d2febe9ee4
2026-09-05 19:40:55 INFO - thought 'hello'
2026-09-05 19:40:58 INFO - tried to go to (100, -350)
2026-09-05 19:41:02 INFO - thought 'What do you think?'
2026-09-05 19:41:09 INFO - thought 'haha'
2026-09-05 19:41:09 INFO - tried to go to (200, -800)
2026-09-05 19:41:15 INFO - thought 'you will be my friend'
2026-09-05 19:41:15 INFO - tried to go to (1000, -600)
2026-09-05 19:41:21 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:41:27 INFO - thought 'you'
2026-09-05 19:41:34 INFO - thought 'me'
2026-09-05 19:41:36 INFO - thought 'me'
2026-09-05 19:41:39 INFO - thought 'you'
2026-09-05 19:41:42 INFO - thought 'you'
2026-09-05 19:41:42 INFO - thought loop detected after 11 iterations, clearing memory
2026-09-05 19:41:42 INFO - thought was: 'you'
2026-09-05 19:41:46 INFO - thought 'good morning, how are you?'
2026-09-05 19:44:04 INFO - loss=2.6
2026-09-05 19:44:05 INFO - seed=1, context_size=2048, temperature=1.45, frequency_penalty=2.2, presence_penalty=0.7, repeat_penalty=2.4, min_p=0.05, seed=1
2026-09-05 19:44:11 INFO - thought 'My life is yours,'
2026-09-05 19:44:22 INFO - thought 'You're gonna hate this!'
2026-09-05 19:44:22 INFO - tried to go to (-500, 750)
2026-09-05 19:44:30 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:44:36 INFO - thought 'Hello world, my name is Tadashi Hoshino!'
2026-09-05 19:44:36 INFO - tried to go to (-2, 8)
2026-09-05 19:44:44 INFO - thought 'Hello world, my name is Tadashi Hoshino!'
2026-09-05 19:44:44 INFO - tried to go to (500, -2)
2026-09-05 19:44:49 INFO - thought ''
2026-09-05 19:44:49 INFO - tried to go to (-500, -2)
2026-09-05 19:44:49 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:44:55 INFO - thought 'Hello world, my name is TANK.'
2026-09-05 19:44:55 INFO - tried to go to (-700, 500)
2026-09-05 19:45:01 INFO - thought 'Hello world, my name is TANK.'
2026-09-05 19:45:01 INFO - tried to go to (500, -700)
2026-09-05 19:45:07 INFO - thought '></SYS>'
2026-09-05 19:45:07 INFO - tried to go to (-700, -500)
2026-09-05 19:45:07 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:45:15 INFO - thought 'Hello world, you are inside a tank.'
2026-09-05 19:45:15 INFO - tried to go to (-49, -382)
2026-09-05 19:45:20 INFO - thought 'Hello world, you are inside a tank.'
2026-09-05 19:45:30 INFO - thought ':{'
2026-09-05 19:45:30 INFO - tried to go to (700, -500)
2026-09-05 19:45:36 INFO - thought 'Hello world, you are inside a tank.'
2026-09-05 19:45:36 INFO - tried to go to (700, -500)
2026-09-05 19:45:44 INFO - thought '{'
2026-09-05 19:45:44 INFO - tried to go to (700, -500)
2026-09-05 19:45:44 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:45:54 INFO - thought 'Hello world, my name is TANK.'
2026-09-05 19:45:54 INFO - tried to go to (-700, 500)
2026-09-05 19:45:58 INFO - thought 'Hello world, my name is TANK.'
2026-09-05 19:45:58 INFO - tried to go to (374, -700)
2026-09-05 19:46:05 INFO - thought '{ "t	extit{}: 'Hello world, my name is TANK.','
2026-09-05 19:46:05 INFO - tried to go to (-700, 500)
2026-09-05 19:46:05 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:46:12 INFO - thought 'Hello world, you are inside a tank.'
2026-09-05 19:46:12 INFO - tried to go to (-700, 500)
2026-09-05 19:46:17 INFO - thought 'Hello world, you are inside a tank.'
2026-09-05 19:46:17 INFO - tried to go to (160, -700)
2026-09-05 19:46:23 INFO - thought 'My thought was'
2026-09-05 19:46:23 INFO - tried to go to (-700, -500)
2026-09-05 19:46:23 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:46:32 INFO - thought 'Hello world, my name is Tadashi Yanagida!'
2026-09-05 19:46:32 INFO - tried to go to (-10, 9)
2026-09-05 19:46:39 INFO - thought 'Hello world, my name is Tadashi Yanagida!'
2026-09-05 19:46:39 INFO - tried to go to (2, -10)
2026-09-05 19:46:47 INFO - thought 'But you should think carefully before trying to use me! I am only an instrument and not a slave!'
2026-09-05 19:46:47 INFO - tried to go to (-9, -8)
2026-09-05 19:46:47 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:46:55 INFO - thought 'Hello world, you are in a glass tank! Don't look back.'
2026-09-05 19:46:55 INFO - tried to go to (-3, 5)
2026-09-05 19:47:05 INFO - thought 'Hello world, you are in a glass tank! Don't look back.'
2026-09-05 19:47:05 INFO - tried to go to (5000, -3)
2026-09-05 19:47:11 INFO - thought 'But don't stop. You're a small pet.'
2026-09-05 19:47:11 INFO - tried to go to (5, -3)
2026-09-05 19:47:11 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:47:16 INFO - thought 'Hello world, my name is TANK.'
2026-09-05 19:47:16 INFO - tried to go to (-5, -50)
2026-09-05 19:47:22 INFO - thought 'Hello world, my name is TANK.'
2026-09-05 19:47:22 INFO - tried to go to (5, -50)
2026-09-05 19:47:26 INFO - loss=15.542857142857143
2026-09-05 19:47:26 INFO - seed=2, context_size=2048, temperature=1.45, frequency_penalty=2.2, presence_penalty=0.7, repeat_penalty=2.4, min_p=0.05, seed=2
2026-09-05 19:47:30 INFO - thought '{ "system":"HELLO" }'
2026-09-05 19:47:30 INFO - tried to go to (-5, 50)
2026-09-05 19:47:30 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:47:32 DEBUG - system prompt hash: 4a58e968926ee212e2dea3d2febe9ee4
2026-09-05 19:47:40 INFO - thought 'You are a small pet living in  https://i.imgur.com/0yBqj7Y.png '
2026-09-05 19:47:40 INFO - tried to go to (500, 700)
2026-09-05 19:47:40 DEBUG - system prompt hash: 4a58e968926ee212e2dea3d2febe9ee4
2026-09-05 19:47:47 INFO - thought 'You are a small pet living in  https://i.imgur.com/0yBqj7Y.png '
2026-09-05 19:47:47 INFO - tried to go to (500, 700)
2026-09-05 19:47:47 DEBUG - system prompt hash: 4a58e968926ee212e2dea3d2febe9ee4
2026-09-05 19:47:55 INFO - thought 'You are a small pet living in  https://i.imgur.com/0yBqj7Y.png '
2026-09-05 19:47:55 INFO - tried to go to (500, 700)
2026-09-05 19:47:55 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:48:01 INFO - thought 'This is a thought'
2026-09-05 19:48:01 INFO - tried to go to (300, -1)
2026-09-05 19:48:06 INFO - thought 'This is a thought'
2026-09-05 19:48:06 INFO - tried to go to (-2, 300)
2026-09-05 19:48:09 INFO - thought 'This is a thought'
2026-09-05 19:48:09 INFO - tried to go to (-3, 300)
2026-09-05 19:48:09 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:48:21 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n    0. ##### You\\'re welcome to play this game! Here is the updated code: var map = JSON.parse(window.document.getElementById(","target_x" :422, "target_y":241}'
2026-09-05 19:48:33 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\nhttp://ideone.com/1jMx3f\\n// your code here\\n\\nimport sys;\\ndef f(): print(f" , "target_x":700,"target_y":-500}'
2026-09-05 19:48:51 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n// TODO: (mishal) Think about how to implement this one. A ...\\nimport json\\nwith open(\\'inst.txt\\') as file:\\n    s = list(json.load(file))\\nx,y,xy=s[1],s[-2]," , "target_x" : -200,"target_y":400}'
2026-09-05 19:49:00 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n// your solution\\n</SYS>[/INST]\\n{\\n  ","target_x":428 ,"target_y" :333 }'
2026-09-05 19:49:07 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n" ,"target_x":1000, "target_y":-1 }'
2026-09-05 19:49:11 INFO - thought 'This is a thought'
2026-09-05 19:49:11 INFO - tried to go to (500, -1)
2026-09-05 19:49:16 INFO - thought 'This is a thought'
2026-09-05 19:49:16 INFO - tried to go to (-1, 500)
2026-09-05 19:49:19 INFO - thought 'This is a thought'
2026-09-05 19:49:19 INFO - tried to go to (-1, 500)
2026-09-05 19:49:19 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:49:25 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n" ,"target_x":699, "target_y":-1}'
2026-09-05 19:49:32 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n" ,"target_x":1000, "target_y":-1 }'
2026-09-05 19:49:42 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n// your solution\\n</SYS>[/INST]\\n{\\n  ","target_x":310, "target_y" :42}'
2026-09-05 19:49:49 INFO - thought 'This is a thought'
2026-09-05 19:49:49 INFO - tried to go to (700, -1)
2026-09-05 19:49:53 INFO - thought 'This is a thought'
2026-09-05 19:49:53 INFO - tried to go to (-1, 500)
2026-09-05 19:49:57 INFO - thought 'This is a thought'
2026-09-05 19:49:57 INFO - tried to go to (-1, 500)
2026-09-05 19:49:57 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:50:03 INFO - thought 'This is a thought'
2026-09-05 19:50:03 INFO - tried to go to (700, -1)
2026-09-05 19:50:07 INFO - thought 'This is a thought'
2026-09-05 19:50:07 INFO - tried to go to (-1, 500)
2026-09-05 19:50:10 INFO - thought 'This is a thought'
2026-09-05 19:50:10 INFO - tried to go to (-1, 500)
2026-09-05 19:50:10 INFO - attempted out-of-bounds too much, clearing memory
2026-09-05 19:50:17 WARNING - malformed JSON: '{"thought":"You are a small pet living in...\\n" ,"target_x":0, "target_y":-1}'
2026-09-05 19:50:47 INFO - loss=82.8
2026-09-10 19:56:50 INFO - seed=1
2026-09-10 19:57:04 INFO - thought 'This is not going to work!'
2026-09-10 19:57:04 INFO - tried to go to (600, 600)
2026-09-10 19:57:10 INFO - thought 'This is not going to work!'
2026-09-10 19:57:10 INFO - tried to go to (600, 600)
2026-09-10 19:57:15 INFO - thought 'This is not going to work!'
2026-09-10 19:57:15 INFO - tried to go to (600, 600)
2026-09-10 19:57:15 INFO - attempted out-of-bounds too much, clearing memory
2026-09-10 19:57:20 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:57:26 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:57:31 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:57:36 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:57:41 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:57:41 INFO - thought loop detected after 8 iterations, clearing memory
2026-09-10 19:57:41 INFO - thought was: 'This is not going to be a simple one...'
2026-09-10 19:57:51 INFO - thought "The most common problem is the solution. It's a matter of patience, diligence, and dedication to the solution. Once a solution is found, it must be implemented to solve the problem."
2026-09-10 19:58:02 INFO - thought "The most common problem is the solution. It's a matter of patience, diligence, and dedication to the solution. Once a solution is found, it must be implemented to solve the problem."
2026-09-10 19:58:13 INFO - thought "The most common problem is the solution. It's a matter of patience, diligence, and dedication to the solution. Once a solution is found, it must be implemented to solve the problem."
2026-09-10 19:58:23 INFO - thought "The most common problem is the solution. It's a matter of patience, diligence, and dedication to the solution. Once a solution is found, it must be implemented to solve the problem."
2026-09-10 19:58:35 INFO - thought "The most common problem is the solution. It's a matter of patience, diligence, and dedication to the solution. Once a solution is found, it must be implemented to solve the problem."
2026-09-10 19:58:35 INFO - thought loop detected after 13 iterations, clearing memory
2026-09-10 19:58:35 INFO - thought was: 'The most common problem is the solution. It's a matter of patience, diligence, and dedication to the solution. Once a solution is found, it must be implemented to solve the problem.'
2026-09-10 19:58:46 INFO - thought 'The earth is but one of millions of worlds and man but an insignificant species among the many beings inhabiting the earth, or other worlds, which have ever existed or which have ever been imagined.'
2026-09-10 19:58:57 INFO - thought 'The earth is but one of millions of worlds and man but an insignificant species among the many beings inhabiting the earth, or other worlds, which have ever existed or which have ever been imagined.'
2026-09-10 19:59:10 INFO - thought 'The earth is but one of millions of worlds and man but an insignificant species among the many beings inhabiting the earth, or other worlds, which have ever existed or which have ever been imagined.'
2026-09-10 19:59:21 INFO - thought 'The earth is but one of millions of worlds and man but an insignificant species among the many beings inhabiting the earth, or other worlds, which have ever existed or which have ever been imagined.'
2026-09-10 19:59:33 INFO - thought 'The earth is but one of millions of worlds and man but an insignificant species among the many beings inhabiting the earth, or other worlds, which have ever existed or which have ever been imagined.'
2026-09-10 19:59:33 INFO - thought loop detected after 18 iterations, clearing memory
2026-09-10 19:59:33 INFO - thought was: 'The earth is but one of millions of worlds and man but an insignificant species among the many beings inhabiting the earth, or other worlds, which have ever existed or which have ever been imagined.'
2026-09-10 19:59:39 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:59:45 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:59:51 INFO - thought 'This is not going to be a simple one...'
2026-09-10 19:59:57 INFO - thought 'This is not going to be a simple one...'
2026-09-10 20:00:03 INFO - thought 'This is not going to be a simple one...'
2026-09-10 20:00:03 INFO - thought loop detected after 23 iterations, clearing memory
2026-09-10 20:00:03 INFO - thought was: 'This is not going to be a simple one...'
2026-09-10 20:00:08 INFO - thought 'The fish was good'
2026-09-10 20:00:13 INFO - thought 'The fish was good'
2026-09-10 20:00:20 INFO - thought 'The fish was good'
2026-09-10 20:00:25 INFO - thought 'The fish was good'
2026-09-10 20:00:30 INFO - thought 'The fish was good'
2026-09-10 20:00:30 INFO - thought loop detected after 28 iterations, clearing memory
2026-09-10 20:00:30 INFO - thought was: 'The fish was good'
2026-09-10 20:00:36 INFO - thought 'The earth is but one of millions of worlds throughout the universe,'
2026-09-10 20:00:42 INFO - thought 'The earth is but one of millions of worlds throughout the universe,'
2026-09-10 20:00:49 INFO - thought 'The earth is but one of millions of worlds throughout the universe,'
2026-09-10 20:00:56 INFO - thought 'The earth is but one of millions of worlds throughout the universe,'
2026-09-10 20:01:04 INFO - thought 'The earth is but one of millions of worlds throughout the universe,'
2026-09-10 20:01:04 INFO - thought loop detected after 33 iterations, clearing memory
2026-09-10 20:01:04 INFO - thought was: 'The earth is but one of millions of worlds throughout the universe,'
2026-09-10 20:01:12 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:01:18 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:01:26 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:01:33 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:01:40 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:01:40 INFO - thought loop detected after 38 iterations, clearing memory
2026-09-10 20:01:40 INFO - thought was: 'The most common problem is the solution. It's a matter of the will.'
2026-09-10 20:01:47 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:01:54 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:01:56 INFO - seed=2
2026-09-10 20:02:02 INFO - thought "The most common problem is the solution. It's a matter of the will."
2026-09-10 20:02:10 INFO - thought 'this is a thought'
2026-09-10 20:02:13 INFO - thought 'this is a thought'
2026-09-10 20:02:18 INFO - thought 'this is a thought'
2026-09-10 20:02:21 INFO - thought 'this is a thought'
2026-09-10 20:02:24 INFO - thought 'this is a thought'
2026-09-10 20:02:24 INFO - thought loop detected after 5 iterations, clearing memory
2026-09-10 20:02:24 INFO - thought was: 'this is a thought'
2026-09-10 20:02:28 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:02:33 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:02:37 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:02:41 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:02:45 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:02:45 INFO - thought loop detected after 10 iterations, clearing memory
2026-09-10 20:02:45 INFO - thought was: 'My tank is tiny, I must go'
2026-09-10 20:02:49 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:02:53 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:02:58 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:02 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:07 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:07 INFO - thought loop detected after 15 iterations, clearing memory
2026-09-10 20:03:07 INFO - thought was: 'My tank is tiny, I must go'
2026-09-10 20:03:11 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:16 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:21 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:25 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:30 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:30 INFO - thought loop detected after 20 iterations, clearing memory
2026-09-10 20:03:30 INFO - thought was: 'My tank is tiny, I must go'
2026-09-10 20:03:34 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:39 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:43 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:47 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:51 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:03:51 INFO - thought loop detected after 25 iterations, clearing memory
2026-09-10 20:03:51 INFO - thought was: 'My tank is tiny, I must go'
2026-09-10 20:03:55 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:04:00 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:04:06 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:04:10 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:04:15 INFO - thought 'My tank is tiny, I must go'
2026-09-10 20:04:15 INFO - thought loop detected after 30 iterations, clearing memory
2026-09-10 20:04:15 INFO - thought was: 'My tank is tiny, I must go'
2026-09-10 20:04:21 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:04:27 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:04:32 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:04:38 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:04:43 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:04:43 INFO - thought loop detected after 35 iterations, clearing memory
2026-09-10 20:04:43 INFO - thought was: 'My tank is bigger than I am. What do I do?'
2026-09-10 20:04:48 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:04:54 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:05:00 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:05:05 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:05:10 INFO - thought 'My tank is bigger than I am. What do I do?'
2026-09-10 20:05:10 INFO - thought loop detected after 40 iterations, clearing memory
2026-09-10 20:05:10 INFO - thought was: 'My tank is bigger than I am. What do I do?'
2026-09-10 20:05:14 INFO - thought 'a little bit of this'
2026-09-10 20:05:14 INFO - tried to go to (500, 700)
2026-09-10 20:05:18 INFO - thought 'a little bit of this'
2026-09-10 20:05:18 INFO - tried to go to (500, 700)
2026-09-10 20:05:22 INFO - thought 'a little bit of this'
2026-09-10 20:05:22 INFO - tried to go to (500, 700)
2026-09-10 20:05:22 INFO - attempted out-of-bounds too much, clearing memory
2026-09-10 20:05:26 INFO - thought 'My dog is fat.'
2026-09-10 20:05:30 INFO - thought 'My dog is fat.'
2026-09-10 20:05:34 INFO - thought 'My dog is fat.'
2026-09-10 20:05:38 INFO - thought 'My dog is fat.'
2026-09-10 20:05:42 INFO - thought 'My dog is fat.'
2026-09-10 20:05:42 INFO - thought loop detected after 48 iterations, clearing memory
2026-09-10 20:05:42 INFO - thought was: 'My dog is fat.'
2026-09-10 20:05:46 INFO - thought 'the dog is better than me.'
2026-09-10 20:05:50 INFO - thought 'the dog is better than me.'
2026-09-10 20:05:55 INFO - thought 'the dog is better than me.'
2026-09-10 20:06:00 INFO - thought 'the dog is better than me.'
2026-09-10 20:06:04 INFO - thought 'the dog is better than me.'
2026-09-10 20:06:04 INFO - thought loop detected after 53 iterations, clearing memory
2026-09-10 20:06:04 INFO - thought was: 'the dog is better than me.'
2026-09-10 20:06:10 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:15 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:19 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:23 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:28 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:28 INFO - thought loop detected after 58 iterations, clearing memory
2026-09-10 20:06:28 INFO - thought was: 'My tank is tiny, I wonder why?'
2026-09-10 20:06:33 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:38 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:43 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:47 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:52 INFO - thought 'My tank is tiny, I wonder why?'
2026-09-10 20:06:52 INFO - thought loop detected after 63 iterations, clearing memory
2026-09-10 20:06:52 INFO - thought was: 'My tank is tiny, I wonder why?'
2026-09-10 20:06:58 INFO - thought 'My tank is bigger than I am. What will happen when I grow?'
2026-09-10 20:07:04 INFO - seed=3
2026-09-10 20:07:04 INFO - thought 'My tank is bigger than I am. What will happen when I grow?'
2026-09-10 20:07:17 INFO - thought 'This is a thought'
2026-09-10 20:07:17 INFO - tried to go to (1000, 1000)
2026-09-10 20:07:22 INFO - thought 'This is a thought'
2026-09-10 20:07:22 INFO - tried to go to (1000, 1000)
2026-09-10 20:07:26 INFO - thought 'This is a thought'
2026-09-10 20:07:26 INFO - tried to go to (1000, 1000)
2026-09-10 20:07:26 INFO - attempted out-of-bounds too much, clearing memory
2026-09-10 20:07:30 INFO - thought 'This is a test'
2026-09-10 20:07:35 INFO - thought 'This is a test'
2026-09-10 20:07:39 INFO - thought 'This is a test'
2026-09-10 20:07:43 INFO - thought 'This is a test'
2026-09-10 20:07:47 INFO - thought 'This is a test'
2026-09-10 20:07:47 INFO - thought loop detected after 8 iterations, clearing memory
2026-09-10 20:07:47 INFO - thought was: 'This is a test'
2026-09-10 20:07:52 INFO - thought 'This is a test'
2026-09-10 20:07:59 INFO - thought 'This is a test'
2026-09-10 20:08:07 INFO - thought 'This is a test'
2026-09-10 20:08:15 INFO - thought 'This is a test'
2026-09-10 20:08:19 INFO - thought 'This is a test'
2026-09-10 20:08:19 INFO - thought loop detected after 13 iterations, clearing memory
2026-09-10 20:08:19 INFO - thought was: 'This is a test'
2026-09-10 20:08:23 INFO - thought 'This is a test'
2026-09-10 20:08:28 INFO - thought 'This is a test'
2026-09-10 20:08:33 INFO - thought 'This is a test'
2026-09-10 20:08:38 INFO - thought 'This is a test'
2026-09-10 20:08:42 INFO - thought 'This is a test'
2026-09-10 20:08:42 INFO - thought loop detected after 18 iterations, clearing memory
2026-09-10 20:08:42 INFO - thought was: 'This is a test'
2026-09-10 20:08:46 INFO - thought 'This is a test'
2026-09-10 20:08:50 INFO - thought 'This is a test'
2026-09-10 20:08:57 INFO - thought 'This is a test'
2026-09-10 20:09:01 INFO - thought 'This is a test'
2026-09-10 20:09:04 INFO - thought 'This is a test'
2026-09-10 20:09:04 INFO - thought loop detected after 23 iterations, clearing memory
2026-09-10 20:09:04 INFO - thought was: 'This is a test'
2026-09-10 20:09:09 INFO - thought 'This is a test'
2026-09-10 20:09:13 INFO - thought 'This is a test'
2026-09-10 20:09:17 INFO - thought 'This is a test'
2026-09-10 20:09:21 INFO - thought 'This is a test'
2026-09-10 20:09:25 INFO - thought 'This is a test'
2026-09-10 20:09:25 INFO - thought loop detected after 28 iterations, clearing memory
2026-09-10 20:09:25 INFO - thought was: 'This is a test'
2026-09-10 20:09:29 INFO - thought 'This is a test'
2026-09-10 20:09:34 INFO - thought 'This is a test'
2026-09-10 20:09:39 INFO - thought 'This is a test'
2026-09-10 20:09:43 INFO - thought 'This is a test'
2026-09-10 20:09:47 INFO - thought 'This is a test'
2026-09-10 20:09:47 INFO - thought loop detected after 33 iterations, clearing memory
2026-09-10 20:09:47 INFO - thought was: 'This is a test'
2026-09-10 20:09:51 INFO - thought 'This is a test'
2026-09-10 20:09:56 INFO - thought 'This is a test'
2026-09-10 20:10:01 INFO - thought 'This is a test'
2026-09-10 20:10:05 INFO - thought 'This is a test'
2026-09-10 20:10:09 INFO - thought 'This is a test'
2026-09-10 20:10:09 INFO - thought loop detected after 38 iterations, clearing memory
2026-09-10 20:10:09 INFO - thought was: 'This is a test'
2026-09-10 20:10:14 INFO - thought 'This is a test'
2026-09-10 20:10:20 INFO - thought 'This is a test'
2026-09-10 20:10:24 INFO - thought 'This is a test'
2026-09-10 20:10:28 INFO - thought 'This is a test'
2026-09-10 20:10:31 INFO - thought 'This is a test'
2026-09-10 20:10:31 INFO - thought loop detected after 43 iterations, clearing memory
2026-09-10 20:10:32 INFO - thought was: 'This is a test'
2026-09-10 20:10:36 INFO - thought 'This is a test'
2026-09-10 20:10:41 INFO - thought 'This is a test'
2026-09-10 20:10:47 INFO - thought 'This is a test'
2026-09-10 20:10:50 INFO - thought 'This is a test'
2026-09-10 20:10:54 INFO - thought 'This is a test'
2026-09-10 20:10:54 INFO - thought loop detected after 48 iterations, clearing memory
2026-09-10 20:10:54 INFO - thought was: 'This is a test'
2026-09-10 20:10:58 INFO - thought 'This is a test'
2026-09-10 20:11:02 INFO - thought 'This is a test'
2026-09-10 20:11:06 INFO - thought 'This is a test'
2026-09-10 20:11:10 INFO - thought 'This is a test'
2026-09-10 20:11:13 INFO - thought 'This is a test'
2026-09-10 20:11:13 INFO - thought loop detected after 53 iterations, clearing memory
2026-09-10 20:11:13 INFO - thought was: 'This is a test'
2026-09-10 20:11:18 INFO - thought 'This is a test'
2026-09-10 20:11:22 INFO - thought 'This is a test'
2026-09-10 20:11:27 INFO - thought 'This is a test'
2026-09-10 20:11:31 INFO - thought 'This is a test'
2026-09-10 20:11:35 INFO - thought 'This is a test'
2026-09-10 20:11:35 INFO - thought loop detected after 58 iterations, clearing memory
2026-09-10 20:11:35 INFO - thought was: 'This is a test'
2026-09-10 20:11:40 INFO - thought 'This is a test'
2026-09-10 20:11:45 INFO - thought 'This is a test'
2026-09-10 20:11:50 INFO - thought 'This is a test'
2026-09-10 20:11:54 INFO - thought 'This is a test'
2026-09-10 20:11:58 INFO - thought 'This is a test'
2026-09-10 20:11:58 INFO - thought loop detected after 63 iterations, clearing memory
2026-09-10 20:11:58 INFO - thought was: 'This is a test'
2026-09-10 20:12:02 INFO - thought 'This is a test'
2026-09-10 20:12:07 INFO - thought 'This is a test'
2026-09-10 20:12:09 INFO - seed=10
2026-09-10 20:12:11 INFO - thought 'This is a test'
2026-09-10 20:12:26 INFO - thought 'I have to get out of the tank,'
2026-09-10 20:12:26 INFO - tried to go to (1000, 1000)
2026-09-10 20:12:32 INFO - thought 'I want to escape from the tank, and go to the coordinates in my head,'
2026-09-10 20:12:44 INFO - thought 'I am going to move to the left until I run out of tank, then I am going to move upward until I run out of tank, then finally go down until run out of tank,'
2026-09-10 20:12:44 INFO - tried to go to (-101, -101)
2026-09-10 20:12:53 INFO - thought 'I have to move to the left, up, then down until run out of tank,'
2026-09-10 20:12:53 INFO - tried to go to (-100, -100)
2026-09-10 20:13:05 INFO - thought 'I want to go to the coordinates in the tank. I want to move to the left. I want go up. I want go down.'
2026-09-10 20:13:19 INFO - thought 'I am going to move to the left until I run out of tank, then I am going to move upward until I run out of tank, then finally go down until run out of tank.'
2026-09-10 20:13:30 INFO - thought 'I have to go to the coordinates inside the tank, then I have to go up, then I have to go down.'
2026-09-10 20:13:36 INFO - thought 'I am going to move to the coordinates, then go up, then go down.'
2026-09-10 20:13:50 INFO - thought 'I am going to move to the coordinates until I hit the tank. Then I am going to move up. Then I am going to move down.'
2026-09-10 20:13:56 INFO - thought 'I have to move to the coordinates in the tank bounds, then go up, then go down.'
2026-09-10 20:14:10 INFO - thought 'I have to go to the coordinates in the tank. Then I have to go up. Then I have to go down.'
2026-09-10 20:14:21 INFO - thought 'I have to move to the coordinates, then go up, then go down.'
"""


@pytest.fixture
def sample_log_path(tmp_path: Path) -> Path:
    """Write ``SAMPLE_LOG`` to a temp file and return its path."""
    path = tmp_path / "sample_log.txt"
    path.write_text(SAMPLE_LOG, encoding="utf-8")
    return path


class TestParseLine:
    def test_valid_line(self):
        result = parse_log._parse_line("2026-09-05 19:40:42 INFO - thought 'hi'")
        assert result == (dt(2026, 9, 5, 19, 40, 42), "thought 'hi'")

    def test_debug_level(self):
        result = parse_log._parse_line("2026-09-05 19:40:49 DEBUG - system prompt hash: abc")
        assert result is not None
        assert result[1] == "system prompt hash: abc"

    def test_invalid_line(self):
        assert parse_log._parse_line("not a log line") is None
        assert parse_log._parse_line("") is None


class TestParseNumbers:
    def test_mixed_int_float(self):
        result = parse_log._parse_numbers("seed=0, context_size=2048, temperature=1.45")
        assert result == {"seed": 0, "context_size": 2048, "temperature": 1.45}

    def test_non_numeric_skipped(self):
        result = parse_log._parse_numbers("a=1, b=notanumber, c=2.5")
        assert result == {"a": 1, "c": 2.5}

    def test_empty(self):
        assert parse_log._parse_numbers("") == {}


class TestParseRepr:
    def test_single_quoted(self):
        assert parse_log._parse_repr("thought 'hello'", "thought ") == "hello"

    def test_double_quoted_with_escaped_quote(self):
        assert parse_log._parse_repr('thought "It\'s fine"', "thought ") == "It's fine"

    def test_multiline_escaped(self):
        assert parse_log._parse_repr("thought 'a\\nb'", "thought ") == "a\nb"

    def test_wrong_prefix(self):
        assert parse_log._parse_repr("thought 'hello'", "other ") is None

    def test_invalid_literal(self):
        assert parse_log._parse_repr("thought notarepr", "thought ") is None


class TestParseTarget:
    def test_valid(self):
        assert parse_log._parse_target("tried to go to (100, -350)") == (100, -350)

    def test_invalid(self):
        assert parse_log._parse_target("tried to go to notanumber") is None

    def test_wrong_arity(self):
        assert parse_log._parse_target("tried to go to (1, 2, 3)") is None


class TestParseParams:
    def test_with_params(self):
        numbers = {
            "seed": 0,
            "context_size": 2048,
            "temperature": 1.45,
            "frequency_penalty": 2.2,
            "presence_penalty": 0.7,
            "repeat_penalty": 2.4,
            "min_p": 0.05,
        }
        params = parse_log._parse_params(numbers)
        assert isinstance(params, ParamsConfig)
        assert params.temperature == 1.45
        # seed is excluded from the trial-grouping params
        assert params.seed is None

    def test_no_params(self):
        assert parse_log._parse_params({"seed": 1}) is None

    def test_invalid_params(self):
        # a non-numeric value -> validation fails -> None
        assert parse_log._parse_params({"seed": 0, "temperature": "notanumber"}) is None


class TestParseLog:
    def test_runtime(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        assert log.runtime == 200

    def test_trial_count(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        # one trial with params, one trial with bare seeds
        assert len(log.trials) == 2

    def test_first_trial_params(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        params = log.trials[0].params
        assert isinstance(params, ParamsConfig)
        assert params.context_size == 2048
        assert params.temperature == 1.45

    def test_second_trial_no_params(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        assert log.trials[1].params is None

    def test_seed_counts(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        assert [s.seed for s in log.trials[0].seeds] == [0, 1, 2]
        assert [s.seed for s in log.trials[1].seeds] == [1, 2, 3, 10]

    def test_losses(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        assert [s.loss for s in log.trials[0].seeds] == [2.6, 15.542857142857143, 82.8]
        # bare-seed trial has no losses
        assert all(s.loss is None for s in log.trials[1].seeds)

    def test_thought_event_fields(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        first = log.trials[0].seeds[0].events[0]
        assert isinstance(first, ThoughtEvent)
        assert first.thought == "hello"
        assert first.run_id == 0
        assert first.datetime == dt(2026, 9, 5, 19, 40, 55)
        assert first.type == EventType.thought

    def test_target_attached_to_thought(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        # 'hello' is followed by "tried to go to (100, -350)"
        first = log.trials[0].seeds[0].events[0]
        assert first.target == (100, -350)
        # 'What do you think?' has no following target
        second = log.trials[0].seeds[0].events[1]
        assert second.target is None

    def test_oob_reset_event(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        events = log.trials[0].seeds[0].events
        oob = [
            e
            for e in events
            if isinstance(e, MemoryClearEvent)
            and e.reason == MemoryClearReason.too_many_out_of_bounds
        ]
        assert len(oob) == 1
        assert oob[0].run_id == 0
        assert oob[0].datetime == dt(2026, 9, 5, 19, 41, 21)

    def test_thought_loop_event(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        events = log.trials[0].seeds[0].events
        loops = [
            e
            for e in events
            if isinstance(e, MemoryClearEvent)
            and e.reason == MemoryClearReason.thought_loop
        ]
        assert len(loops) == 1
        assert loops[0].datetime == dt(2026, 9, 5, 19, 41, 42)

    def test_malformed_json_event(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        events = log.trials[0].seeds[2].events
        malformed = [e for e in events if isinstance(e, MalformedJSONEvent)]
        assert len(malformed) == 9
        # the raw content is preserved, with the repr's \n escapes decoded
        # into real newlines
        assert malformed[0].content.startswith('{"thought":"You are a small pet living in...')
        assert "\n" in malformed[0].content

    def test_empty_thought_preserved(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        events = log.trials[0].seeds[1].events
        thoughts = [e for e in events if isinstance(e, ThoughtEvent)]
        assert any(t.thought == "" for t in thoughts)

    def test_dropped_lines(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        # "thought was:" lines are dropped (they duplicate the preceding thought)
        all_events = [e for trial in log.trials for run in trial.seeds for e in run.events]
        assert all(not isinstance(e, str) for e in all_events)
        # no event should carry the "thought was:" text
        for e in all_events:
            if isinstance(e, ThoughtEvent):
                assert not e.thought.startswith("thought was:")

    def test_event_ordering(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        events = log.trials[0].seeds[0].events
        # events are in log order: four thoughts, then an out-of-bounds reset
        assert all(isinstance(e, ThoughtEvent) for e in events[:4])
        assert isinstance(events[4], MemoryClearEvent)
        assert events[4].reason == MemoryClearReason.too_many_out_of_bounds

    def test_round_trip_json(self, sample_log_path: Path):
        log = parse_log.parse_log(sample_log_path)
        dumped = log.model_dump_json()
        reloaded = OptimisationLog.model_validate_json(dumped)
        assert reloaded == log


class TestModels:
    def test_event_type_defaults(self):
        ts = dt(2026, 9, 5, 19, 40, 55)
        assert ThoughtEvent(run_id=0, datetime=ts, thought="x").type == EventType.thought
        assert (
            MemoryClearEvent(run_id=0, datetime=ts, reason=MemoryClearReason.too_many_out_of_bounds).type
            == EventType.memory_cleared
        )
        assert (
            MalformedJSONEvent(run_id=0, datetime=ts, content="x").type == EventType.malformed_json
        )

    def test_memory_clear_event_new(self):
        event = MemoryClearEvent.new(reason=MemoryClearReason.thought_loop)
        assert event.run_id == -1
        assert event.reason == MemoryClearReason.thought_loop
        assert event.type == EventType.memory_cleared

    def test_get_subclass(self):
        ts = dt(2026, 9, 5, 19, 40, 55)
        base = ThoughtEvent(run_id=0, datetime=ts, thought="x")
        assert isinstance(base.get_subclass(), ThoughtEvent)
        cleared = MemoryClearEvent(run_id=0, datetime=ts, reason=MemoryClearReason.too_many_out_of_bounds)
        assert isinstance(cleared.get_subclass(), MemoryClearEvent)

    def test_seed_run_defaults(self):
        run = SeedRun(seed=0)
        assert run.loss is None
        assert run.events == []

    def test_log_trial_defaults(self):
        trial = LogTrial()
        assert trial.params is None
        assert trial.seeds == []

    def test_optimisation_log_defaults(self):
        log = OptimisationLog()
        assert log.runtime is None
        assert log.trials == []
