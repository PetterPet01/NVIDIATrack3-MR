import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

def infer_once(server_url: str, file_path: str):
    try:
        with open(file_path, "rb") as f:
            start = time.time()
            response = requests.post(f"{server_url}/infer/", files={"file": ("input.json", f)})
            latency = time.time() - start
            return latency, response.status_code
    except Exception as e:
        return -1, str(e)

def run_benchmark(server_url: str, file_path: str, n_requests: int, concurrency: int):
    print(f"🚀 Running benchmark: {n_requests} requests | {concurrency} concurrent threads")
    latencies = []
    success = 0

    start_all = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(infer_once, server_url, file_path) for _ in range(n_requests)]
        for future in as_completed(futures):
            latency, status = future.result()
            if status == 200:
                latencies.append(latency)
                success += 1
            else:
                print("❌ Failed:", status)
    end_all = time.time()

    total_time = end_all - start_all

    if latencies:
        print(f"\n✅ Benchmark Results")
        print(f"  Successful:     {success}/{n_requests}")
        print(f"  Total Time:     {total_time:.2f} s")
        print(f"  Throughput:     {success / total_time:.2f} requests/sec")
        print(f"  Avg Latency:    {sum(latencies)/len(latencies):.2f} s")
        print(f"  Min Latency:    {min(latencies):.2f} s")
        print(f"  Max Latency:    {max(latencies):.2f} s")
    else:
        print("❌ No successful responses.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://localhost:9000", help="URL of the UniK3D inference server")
    parser.add_argument("--file", type=str, default="/workspace/PhysicalAI_Dataset/train_sample/demo.json", help="Path to a test JSON input file")
    parser.add_argument("--n", type=int, default=10, help="Total number of requests")
    parser.add_argument("--c", type=int, default=1, help="Concurrent threads")

    args = parser.parse_args()
    run_benchmark(args.url, args.file, args.n, args.c)
