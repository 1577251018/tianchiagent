import json
import time
from main import run_agent

# ==========================================
# 用户需实现：你的智能搜索 agent 调用函数
# ==========================================
def search_agent(question: str) -> str:
    """
    这里模拟调用你的智能搜索 agent。
    请替换为实际的调用逻辑，例如 API 请求、模型推理等。
    """
    # 示例：模拟搜索延迟并返回一个假答案
    # time.sleep(0.5)
    # return f"【模拟答案】关于“{question}”的搜索结果。"
    
    # TODO: 替换为你的实际 agent 调用逻辑
    try:
        # 示例：假设你有一个 Agent 类或函数
        # from my_agent import MySearchAgent
        return run_agent(question)
    except Exception as e:
        return f"[Error] {str(e)}"


# ==========================================
# 批量测试主函数
# ==========================================
def run_batch_test(input_file: str, output_file: str):
    """
    读取 JSONL 测试题，调用 agent，保存结果
    """
    print(f"开始批量测试，读取文件: {input_file}")
    
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:

        total = 0
        start_time = time.time()

        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                item = json.loads(line)
                qid = item.get("id")
                question = item.get("question", "").strip()

                if not question:
                    answer = "[Empty question]"
                    time_spent = 0.0
                else:
                    # 记录单题耗时
                    q_start = time.time()
                    answer = search_agent(question)
                    time_spent = time.time() - q_start

                # 构造结果
                result = {
                    "id": qid,
                    "question": question,
                    "answer": answer,
                    "time_spent": round(time_spent, 4)
                }

                # 写入结果文件
                fout.write(json.dumps(result, ensure_ascii=False) + '\n')
                fout.flush()  # 确保实时写入，避免中断丢失数据

                # 打印进度
                print(f"[{total}] ID: {qid} | Q: {question[:50]}... | Time: {time_spent:.2f}s")

            except Exception as e:
                print(f"[Error] 处理第 {total} 行失败: {e}")
                continue

        total_time = time.time() - start_time
        print(f" 测试完成！共处理 {total} 道题目，总耗时: {total_time:.2f} 秒")
        print(f"结果已保存至: {output_file}")


# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    input_path = r"C:\Users\whr\Desktop\my_test\question.jsonl"    # 输入：测试题文件
    output_path = r"C:\Users\whr\Desktop\my_test\results.jsonl"   # 输出：测试结果文件


    run_batch_test(input_path, output_path)