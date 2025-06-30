import re
import json
from typing import Union, Tuple, List, Dict

class TextEvaluator:
    """
    Hệ thống đánh giá văn bản với 4 loại category:
    - count: So sánh số với độ lệch ±10%
    - distance: So sánh số với độ lệch ±10%
    - mcq: So sánh số phải hoàn toàn giống nhau
    - left_right: So sánh text phải hoàn toàn giống nhau (left/right)
    """
    
    def __init__(self):
        self.valid_categories = ['count', 'distance', 'mcq', 'left_right']
        self.valid_left_right = ['left', 'right']
    
    def extract_number(self, text: str) -> float:
        """
        Trích xuất số từ text
        
        Args:
            text (str): Text chứa số
            
        Returns:
            float: Số được trích xuất
            
        Raises:
            ValueError: Nếu không tìm thấy số trong text
        """
        # Loại bỏ khoảng trắng và chuyển về lowercase
        text = str(text).strip().lower()
        
        # Tìm tất cả các số (bao gồm số thập phân)
        numbers = re.findall(r'-?\d+\.?\d*', text)
        
        if not numbers:
            raise ValueError(f"Không tìm thấy số trong text: '{text}'")
        
        # Lấy số đầu tiên tìm được
        return float(numbers[0])
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch text cho việc so sánh
        
        Args:
            text (str): Text cần làm sạch
            
        Returns:
            str: Text đã được làm sạch
        """
        return str(text).strip().lower()
    
    def evaluate_count_distance(self, output_text: str, ground_truth: str) -> Tuple[bool, str]:
        """
        Đánh giá cho category 'count' và 'distance'
        Chấp nhận nếu độ lệch nằm trong khoảng ±10%
        
        Args:
            output_text (str): Text output cần đánh giá
            ground_truth (str): Text ground truth
            
        Returns:
            Tuple[bool, str]: (Kết quả đúng/sai, Thông tin chi tiết)
        """
        try:
            output_num = self.extract_number(output_text)
            truth_num = self.extract_number(ground_truth)
            
            # Tính độ lệch phần trăm
            if truth_num == 0:
                # Nếu ground truth = 0, chỉ chấp nhận output = 0
                is_correct = (output_num == 0)
                detail = f"Output: {output_num}, Ground Truth: {truth_num}"
            else:
                deviation_percent = abs((output_num - truth_num) / truth_num) * 100
                is_correct = deviation_percent <= 10.0
                detail = f"Output: {output_num}, Ground Truth: {truth_num}, Độ lệch: {deviation_percent:.2f}%"
            
            return is_correct, detail
            
        except ValueError as e:
            return False, f"Lỗi chuyển đổi số: {str(e)}"
    
    def evaluate_mcq(self, output_text: str, ground_truth: str) -> Tuple[bool, str]:
        """
        Đánh giá cho category 'mcq'
        Chỉ chấp nhận khi 2 số hoàn toàn giống nhau
        
        Args:
            output_text (str): Text output cần đánh giá
            ground_truth (str): Text ground truth
            
        Returns:
            Tuple[bool, str]: (Kết quả đúng/sai, Thông tin chi tiết)
        """
        try:
            output_num = self.extract_number(output_text)
            truth_num = self.extract_number(ground_truth)
            
            is_correct = (output_num == truth_num)
            detail = f"Output: {output_num}, Ground Truth: {truth_num}"
            
            return is_correct, detail
            
        except ValueError as e:
            return False, f"Lỗi chuyển đổi số: {str(e)}"
    
    def evaluate_left_right(self, output_text: str, ground_truth: str) -> Tuple[bool, str]:
        """
        Đánh giá cho category 'left_right'
        Chỉ chấp nhận 'left' hoặc 'right' và phải hoàn toàn giống nhau
        
        Args:
            output_text (str): Text output cần đánh giá
            ground_truth (str): Text ground truth
            
        Returns:
            Tuple[bool, str]: (Kết quả đúng/sai, Thông tin chi tiết)
        """
        output_clean = self.clean_text(output_text)
        truth_clean = self.clean_text(ground_truth)
        
        # Kiểm tra xem cả hai có phải là left/right không
        if output_clean not in self.valid_left_right:
            return False, f"Output không hợp lệ: '{output_text}' (chỉ chấp nhận 'left' hoặc 'right')"
        
        if truth_clean not in self.valid_left_right:
            return False, f"Ground Truth không hợp lệ: '{ground_truth}' (chỉ chấp nhận 'left' hoặc 'right')"
        
        is_correct = (output_clean == truth_clean)
        detail = f"Output: '{output_clean}', Ground Truth: '{truth_clean}'"
        
        return is_correct, detail
    
    def evaluate(self, output_text: str, ground_truth: str, category: str) -> dict:
        """
        Hàm chính để đánh giá
        
        Args:
            output_text (str): Text output cần đánh giá
            ground_truth (str): Text ground truth
            category (str): Loại đánh giá ('count', 'distance', 'mcq', 'left_right')
            
        Returns:
            dict: Kết quả đánh giá với các thông tin chi tiết
        """
        # Kiểm tra category hợp lệ
        if category not in self.valid_categories:
            return {
                'is_correct': False,
                'category': category,
                'error': f"Category không hợp lệ. Chỉ chấp nhận: {self.valid_categories}",
                'detail': ''
            }
        
        # Thực hiện đánh giá theo category
        if category in ['count', 'distance']:
            is_correct, detail = self.evaluate_count_distance(output_text, ground_truth)
        elif category == 'mcq':
            is_correct, detail = self.evaluate_mcq(output_text, ground_truth)
        elif category == 'left_right':
            is_correct, detail = self.evaluate_left_right(output_text, ground_truth)
        
        return {
            'is_correct': is_correct,
            'category': category,
            'output_text': output_text,
            'ground_truth': ground_truth,
            'detail': detail,
            'error': None if is_correct else 'Đánh giá không chính xác'
        }

    def evaluate_from_json(self, json_file_path: str) -> Dict:
        """
        Đánh giá từ file JSON
        
        Args:
            json_file_path (str): Đường dẫn đến file JSON
            
        Returns:
            Dict: Kết quả đánh giá tổng thể
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            return {
                'error': f"Không tìm thấy file: {json_file_path}",
                'total_score': 0.0,
                'results': []
            }
        except json.JSONDecodeError as e:
            return {
                'error': f"Lỗi định dạng JSON: {str(e)}",
                'total_score': 0.0,
                'results': []
            }
        
        return self.evaluate_from_data(data)
    
    def evaluate_from_data(self, data: List[Dict]) -> Dict:
        """
        Đánh giá từ dữ liệu đã load
        
        Args:
            data (List[Dict]): Danh sách các mục cần đánh giá
            
        Returns:
            Dict: Kết quả đánh giá tổng thể
        """
        if not isinstance(data, list):
            return {
                'error': 'Dữ liệu phải là một danh sách (array)',
                'total_score': 0.0,
                'results': []
            }
        
        results = []
        correct_count = 0
        total_count = len(data)
        
        for i, item in enumerate(data):
            # Kiểm tra format của từng item
            if not self._validate_item_format(item):
                result = {
                    'index': i,
                    'image': item.get('image', 'N/A'),
                    'category': item.get('category', 'N/A'),
                    'is_correct': False,
                    'error': 'Format không hợp lệ. Cần có: image, category, output_text, ground_true',
                    'detail': ''
                }
            else:
                # Thực hiện đánh giá
                eval_result = self.evaluate(
                    item['output_text'], 
                    item['ground_true'], 
                    item['category']
                )
                
                result = {
                    'index': i,
                    'image': item['image'],
                    'category': item['category'],
                    'output_text': item['output_text'],
                    'ground_true': item['ground_true'],
                    'is_correct': eval_result['is_correct'],
                    'detail': eval_result['detail'],
                    'error': eval_result['error']
                }
                
                if eval_result['is_correct']:
                    correct_count += 1
            
            results.append(result)
        
        # Tính điểm cuối cùng
        final_score = correct_count / total_count if total_count > 0 else 0.0
        
        return {
            'total_items': total_count,
            'correct_items': correct_count,
            'incorrect_items': total_count - correct_count,
            'total_score': final_score,
            'results': results,
            'error': None
        }
    
    def _validate_item_format(self, item: Dict) -> bool:
        """
        Kiểm tra format của một item
        
        Args:
            item (Dict): Item cần kiểm tra
            
        Returns:
            bool: True nếu format hợp lệ
        """
        required_fields = ['image', 'category', 'output_text', 'ground_true']
        return all(field in item for field in required_fields)
    
    def print_summary(self, evaluation_result: Dict) -> None:
        """
        In tóm tắt kết quả đánh giá
        
        Args:
            evaluation_result (Dict): Kết quả từ evaluate_from_json hoặc evaluate_from_data
        """
        if evaluation_result.get('error'):
            print(f"- LỖI: {evaluation_result['error']}")
            return
        
        print("=" * 60)
        print("KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ")
        print("=" * 60)
        print(f"- Tổng số mục: {evaluation_result['total_items']}")
        print(f"- Số mục đúng: {evaluation_result['correct_items']}")
        print(f"- Số mục sai: {evaluation_result['incorrect_items']}")
        print(f"- ĐIỂM SỐ: {evaluation_result['total_score']:.4f} / 1.0000")
        print(f"- TỈ LỆ CHÍNH XÁC: {evaluation_result['total_score']*100:.2f}%")
        print("=" * 60)
    
    def print_detailed_results(self, evaluation_result: Dict, show_correct: bool = False) -> None:
        """
        In kết quả chi tiết
        
        Args:
            evaluation_result (Dict): Kết quả đánh giá
            show_correct (bool): Có hiển thị cả kết quả đúng không
        """
        if evaluation_result.get('error'):
            return
        
        print("\nCHI TIẾT KẾT QUẢ:")
        print("-" * 60)
        
        # Thống kê theo category
        category_stats = {}
        for result in evaluation_result['results']:
            category = result['category']
            if category not in category_stats:
                category_stats[category] = {'correct': 0, 'total': 0}
            
            category_stats[category]['total'] += 1
            if result['is_correct']:
                category_stats[category]['correct'] += 1
        
        print("Thống kê theo Category:")
        for category, stats in category_stats.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {category}: {stats['correct']}/{stats['total']} ({accuracy*100:.1f}%)")
        
        print("-" * 60)
        
        # Chi tiết từng mục
        for result in evaluation_result['results']:
            if not show_correct and result['is_correct']:
                continue
                
            status = " ĐÚNG" if result['is_correct'] else "❌ SAI"
            print(f"#{result['index']+1} - {status}")
            print(f"   Image: {result['image']}")
            print(f"   Category: {result['category']}")
            
            if not result['is_correct']:
                print(f"   Output: {result.get('output_text', 'N/A')}")
                print(f"   Ground Truth: {result.get('ground_true', 'N/A')}")
                print(f"   Chi tiết: {result['detail']}")
                if result['error']:
                    print(f"    Lỗi: {result['error']}")
            
            print("-" * 40)

# Hàm tiện ích để đánh giá từ file JSON
def evaluate_json_file(json_file_path: str, show_details: bool = False, show_correct: bool = False) -> float:
    """
    Hàm tiện ích để đánh giá file JSON và in kết quả
    
    Args:
        json_file_path (str): Đường dẫn đến file JSON
        show_details (bool): Có hiển thị chi tiết không
        show_correct (bool): Có hiển thị cả kết quả đúng không
        
    Returns:
        float: Điểm số từ 0.0 đến 1.0
    """
    evaluator = TextEvaluator()
    result = evaluator.evaluate_from_json(json_file_path)
    
    # In tóm tắt
    evaluator.print_summary(result)
    
    # In chi tiết nếu được yêu cầu
    if show_details:
        evaluator.print_detailed_results(result, show_correct)
    
    return result.get('total_score', 0.0)


# Hàm tiện ích để sử dụng nhanh (đánh giá đơn lẻ)
def evaluate_text(output_text: str, ground_truth: str, category: str) -> dict:
    """
    Hàm tiện ích để đánh giá nhanh một mục đơn lẻ
    
    Args:
        output_text (str): Text output cần đánh giá
        ground_truth (str): Text ground truth
        category (str): Loại đánh giá
        
    Returns:
        dict: Kết quả đánh giã
    """
    evaluator = TextEvaluator()
    return evaluator.evaluate(output_text, ground_truth, category)

def create_sample_json(filename: str = 'sample_evaluation.json') -> None:
    """
    Tạo file JSON mẫu để test
    
    Args:
        filename (str): Tên file JSON
    """
    sample_data = [
        {
            "image": "count_example.jpg",
            "category": "count", 
            "output_text": "I can see 13 objects in total",
            "ground_true": "12"
        },
        {
            "image": "distance_measurement.jpg",
            "category": "distance",
            "output_text": "100",
            "ground_true": "98"
        },
        {
            "image": "multiple_choice_q1.jpg", 
            "category": "mcq",
            "output_text": "region3",
            "ground_true": "3"
        },
        {
            "image": "direction_arrow.jpg",
            "category": "left_right",
            "output_text": "left",
            "ground_true": "left"
        },
        {
            "image": "another_direction.jpg",
            "category": "left_right",
            "output_text": "Right",
            "ground_true": "right"
        },
        {
            "image": "count_objects2.jpg",
            "category": "count",
            "output_text": "20",
            "ground_true": "20"  # 20% difference - should be incorrect
        }
    ]
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"Đã tạo file mẫu: {filename}")

# Ví dụ sử dụng
if __name__ == "__main__":
    
    # Tạo file JSON mẫu
    create_sample_json()
    
    # print("\n🔍 Bắt đầu đánh giá từ file JSON...")
    # print("-"*60)
    
    # Đánh giá từ file JSON với báo cáo đầy đủ
    final_score = evaluate_json_file('sample_evaluation.json', show_details=True, show_correct=False)
    
    print(f"\n ĐIỂM CUỐI CÙNG: {final_score:.4f} / 1.0000")
    print(f" TỶ LỆ CHÍNH XÁC: {final_score*100:.2f}%")
    
    # # Ví dụ sử dụng trực tiếp evaluator
    # print("\n" + "="*60)
    # print("🧪 VÍ DỤ SỬ DỤNG TRỰC TIẾP")
    # print("="*60)
    
    # evaluator = TextEvaluator()
    
    # Test một số trường hợp cụ thể
    # test_cases = [
    #     ("The answer is 100", "95", "count", "✅ Trong khoảng ±10%"),
    #     ("50", "40", "distance", "❌ Ngoài khoảng ±10%"), 
    #     ("Option A: 2", "2", "mcq", "✅ Số hoàn toàn giống nhau"),
    #     ("left side", "right", "left_right", "❌ Khác nhau"),
    #     ("RIGHT", "right", "left_right", "✅ Case insensitive")
    # ]
    
    # for i, (output, truth, category, expected) in enumerate(test_cases, 1):
    #     result = evaluator.evaluate(output, truth, category)
    #     status = "✅ ĐÚNG" if result['is_correct'] else "❌ SAI"
        
    #     print(f"\n#{i} - {category.upper()}")
    #     print(f"   Output: '{output}'")
    #     print(f"   Ground Truth: '{truth}'")
    #     print(f"   Kết quả: {status}")
    #     print(f"   Chi tiết: {result['detail']}")
    #     print(f"   Mong đợi: {expected}")
        
    # print("\n" + "="*60)
    # print("📋 HƯỚNG DẪN SỬ DỤNG:")
    # print("="*60)
    # print("1. Chuẩn bị file JSON với format:")
    # print("   [{'image': '...', 'category': '...', 'output_text': '...', 'ground_true': '...'}]")
    # print("")
    # print("2. Sử dụng hàm evaluate_json_file():")
    # print("   score = evaluate_json_file('your_file.json', show_details=True)")
    # print("")
    # print("3. Hoặc sử dụng trực tiếp TextEvaluator:")
    # print("   evaluator = TextEvaluator()")
    # print("   result = evaluator.evaluate_from_json('your_file.json')")
    # print("   evaluator.print_summary(result)")
    # print("\n🎉 Hoàn thành!")