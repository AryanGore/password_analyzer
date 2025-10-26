import random

def reservoir_sample(input_path, output_path, k=100000, encoding='latin-1'):
    reservoir = []
    with open(input_path, 'r', encoding=encoding, errors='ignore') as f:
        for i, line in enumerate(f):
            pw = line.strip()
            if not pw:
                continue
            if len(reservoir) < k:
                reservoir.append(pw)
            else:
                j = random.randint(0, i)
                if j < k:
                    reservoir[j] = pw

    with open(output_path, 'w', encoding=encoding) as out:
        for pw in reservoir:
            out.write(pw + '\n')

if __name__ == "__main__":
    reservoir_sample("rockyou.txt", "rockyou_100k.txt", k=100000)
    print("Done: rockyou_100k.txt created (reservoir sampling).")
