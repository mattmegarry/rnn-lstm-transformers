
from char_tokenizer import CharTokenizer

tokenizer = CharTokenizer()

""" def char_generate(chars):
    char_list = list(chars)
    x = torch.cat([sos, torch.tensor(tokenizer.encode(char_list))])
  while True:
    p, _ = m(x)
    p = torch.nn.functional.softmax(p, dim=1)
    p = torch.argmax(p, dim=1)
    x = torch.cat([x, p[-1].unsqueeze(0)])
    if p[-1] == 1 or len(p.tolist()) == 17: break
  print("Generate:", tokenizer.decode(x.tolist())) """