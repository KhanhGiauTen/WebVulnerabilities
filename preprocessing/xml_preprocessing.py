import re
import pandas as pd


def parse_single_request(text):
    lines = text.strip().split('\n')
    data = {}

    # Lấy label (class)
    for line in lines:
        if line.lower().startswith("class:"):
            data['classification'] = line.split(":", 1)[1].strip()

    # Lấy Method và URL
    for line in lines:
        match = re.match(r'^(GET|POST|PUT|DELETE|HEAD|OPTIONS|PATCH)\s+([^\s]+)', line)
        if match:
            data['Method'] = match.group(1)
            data['URL'] = match.group(2)
            break

    # Lấy các HTTP header cần thiết
    fields = ['Host', 'User-Agent', 'Accept', 'Content-Type', 'Content', 'Cookie']
    for field in fields:
        for line in lines:
            if line.lower().startswith(field.lower() + ":"):
                data[field] = line.split(":", 1)[1].strip()
                break
        if field not in data:
            data[field] = None  # Bổ sung nếu không có

    return data

# Đọc từ file chứa nhiều request (giả sử mỗi request cách nhau bằng dòng '----:')
with open("C:/Users/Danny Phong/Downloads/dataset_ecml_pkdd_train_test/xml_train.txt", "r", encoding="utf-8") as f:
    content = f.read()

requests = content.split("----:")  # Tách các request
parsed_data = [parse_single_request(req) for req in requests if req.strip()]

df = pd.DataFrame(parsed_data)
df.to_csv("parsed_requests_train.csv", index=False)
print("✅ Dữ liệu đã được chuyển thành file CSV: parsed_requests.csv")