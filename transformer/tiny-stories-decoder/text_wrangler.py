import re
import os

num_stories = 2048
count = 0

def transmute_stories(num_stories=num_stories, count=count):
    os.remove('TinyStories-2048.txt')
    with open('TinyStories-train.txt') as source, open('TinyStories-2048.txt', 'a') as target:
        line_accumulator = ""
        for line in source:
            if num_stories <= count:
                return
            line = re.sub(r"\n", " ", line)
            if re.search("<|endoftext|>", line) is None:
                line_accumulator += line    
            else:
                line_accumulator = line_accumulator.strip()
                print(line_accumulator, file=target, end='\n')
                line_accumulator = ""
                count += 1

transmute_stories(num_stories, count)

target_check = 0
with open('TinyStories-2048.txt') as target:
    for line in target:
        target_check += 1
print("Stories written: ", target_check)       