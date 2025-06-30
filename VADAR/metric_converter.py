import os
import cv2
import numpy as np
from PIL import Image
import torch
from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt # Not strictly needed for JSON generation
import json

# --- UniK3D Model Loading ---
try:
    from unik3d.models import UniK3D
    UNIK3D_AVAILABLE = True
except ImportError:
    print("UniK3D library not found. Please install it using: pip install unik3d")
    UNIK3D_AVAILABLE = False

def instantiate_unik3d_model(model_type="Large"):
    type_ = model_type[0].lower()
    name = f"unik3d-vit{type_}"
    try:
        model = UniK3D.from_pretrained(f"lpiccinelli/{name}")
    except Exception as e:
        print(f"Lỗi khi tải UniK3D: {e}")
        raise
    model.resolution_level = 9
    model.interpolation_mode = "bilinear"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"UniK3D '{name}' đã tải lên: {device}")
    return model, device

# --- Các hàm xử lý ảnh và độ sâu (largely unchanged from your provided code) ---

def load_direct_relative_depth_gt(png_path, target_shape_hw=None):
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"File Relative Depth GT PNG không tìm thấy: {png_path}")

    try:
        img_pil = Image.open(png_path)
        depth_raw_data = np.array(img_pil)

        if depth_raw_data.ndim == 3:
            depth_raw_data = depth_raw_data[:,:,0]

        relative_depth_gt = None
        if depth_raw_data.dtype == np.float32:
            min_val, max_val = np.nanmin(depth_raw_data), np.nanmax(depth_raw_data)
            if 0 <= min_val <= max_val <= 1.001: # Allow small tolerance
                relative_depth_gt = depth_raw_data
            else:
                if max_val > min_val:
                    relative_depth_gt = (depth_raw_data - min_val) / (max_val - min_val)
                else: # All values are same or only one distinct value
                    relative_depth_gt = np.zeros_like(depth_raw_data, dtype=np.float32)
        elif depth_raw_data.dtype == np.uint8:
            relative_depth_gt = depth_raw_data.astype(np.float32) / 255.0
        elif depth_raw_data.dtype == np.uint16:
            relative_depth_gt = depth_raw_data.astype(np.float32) / 65535.0
        elif img_pil.mode == 'I' and depth_raw_data.dtype == np.int32:
            # print(f"Warning: Interpreting 'I' mode (int32) depth map from {png_path}. Assuming values are scaled to 65535 for relative depth.")
            max_possible_val = np.iinfo(np.uint16).max
            # if np.any(depth_raw_data < 0) :
            #     print(f"Warning: int32 depth map {png_path} contains negative values. Clipping to 0 for relative depth.")
            #     depth_raw_data = np.clip(depth_raw_data, 0, None)

            # if np.max(depth_raw_data) > max_possible_val * 1.1: # Allow some leeway if it's scaled differently
            #      print(f"Warning: Max value in int32 depth map ({np.max(depth_raw_data)}) significantly exceeds {max_possible_val}. Relative depth might be inaccurate if assumption is wrong.")
            
            # A more robust way for unknown int32 scaling might be min-max normalization if it's not already relative
            # For now, sticking to the user's original logic but ensuring positive values for division
            # if np.max(depth_raw_data) > 0:
            #    relative_depth_gt = depth_raw_data.astype(np.float32) / np.max(depth_raw_data) # Normalize by actual max
            # else:
            #    relative_depth_gt = np.zeros_like(depth_raw_data, dtype=np.float32)
            # Reverting to user's original logic for 'I' mode for now, as changing it might be unintended
            relative_depth_gt = depth_raw_data.astype(np.float32) / max_possible_val

        else:
            raise ValueError(f"Không thể diễn giải file Relative Depth GT với mode '{img_pil.mode}' và dtype '{depth_raw_data.dtype}' thành dạng 0-1 một cách tự động.")

        relative_depth_gt[np.isnan(relative_depth_gt)] = 0 # Replace NaNs from division by zero if min_val=max_val
        relative_depth_gt = np.clip(relative_depth_gt, 0, 1)

        if target_shape_hw is not None and relative_depth_gt.shape != target_shape_hw:
            relative_depth_gt = cv2.resize(relative_depth_gt, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_LINEAR)

        return relative_depth_gt
    except Exception as e:
        # print(f"Lỗi nghiêm trọng khi mở và xử lý file Relative Depth GT {png_path}: {e}")
        raise

def create_unik3d_metric_depth(rgb_image_path, unik3d_model, device, target_height=480):
    if not os.path.exists(rgb_image_path):
        raise FileNotFoundError(f"Ảnh RGB không tìm thấy: {rgb_image_path}")
    image_pil = Image.open(rgb_image_path).convert("RGB")
    w_orig, h_orig = image_pil.size
    if h_orig == 0: raise ValueError("Chiều cao ảnh RGB bằng 0.")
    scale = target_height / h_orig
    target_width = int(w_orig * scale)
    if target_width == 0: raise ValueError("Chiều rộng ảnh RGB sau resize bằng 0.")
    image_pil_resized = image_pil.resize((target_width, target_height), Image.LANCZOS)
    image_rgb_np = np.array(image_pil_resized)
    image_tensor = torch.from_numpy(image_rgb_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad(): outputs = unik3d_model.infer(image_tensor, camera=None, normalize=True)
    m_md_u = None
    if 'depth' in outputs and outputs['depth'] is not None:
        depth_tensor = outputs['depth']
        if depth_tensor.ndim == 4 and depth_tensor.shape[0:2] == (1,1):
            m_md_u = depth_tensor.squeeze().cpu().numpy()
            if m_md_u.ndim != 2: m_md_u = None
    if m_md_u is None and 'points' in outputs and outputs['points'] is not None:
        points_tensor = outputs["points"]
        if points_tensor.ndim == 4 and points_tensor.shape[0:2] == (1,3):
            points_3d = points_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
            if points_3d.ndim == 3 and points_3d.shape[2] == 3:
                m_md_u = points_3d[:, :, 2]
    if m_md_u is None: raise ValueError("Không thể trích xuất metric depth 2D từ UniK3D.")
    m_md_u[np.isinf(m_md_u)] = np.nan # Replace inf with nan early
    m_md_u[m_md_u <= 1e-3] = np.nan
    return m_md_u, m_md_u.shape

def fit_linear_and_refine_metric_depth(M_MD_u, M_RD, min_rd_thresh=0.01, min_points_for_fit=100):
    if M_MD_u.shape != M_RD.shape:
        # Try to resize M_RD if shapes don't match, common if M_MD_u is from a model with fixed output size
        # print(f"Warning: Shape mismatch in fit_linear_and_refine_metric_depth. M_MD_u: {M_MD_u.shape}, M_RD: {M_RD.shape}. Resizing M_RD.")
        M_RD_resized = cv2.resize(M_RD, (M_MD_u.shape[1], M_MD_u.shape[0]), interpolation=cv2.INTER_LINEAR)
        if M_RD_resized.shape != M_MD_u.shape: # Check if resize was successful
             raise ValueError(f"Shape mismatch persists after attempting to resize M_RD: M_MD_u {M_MD_u.shape} vs M_RD_resized {M_RD_resized.shape}")
        M_RD_to_use = M_RD_resized
    else:
        M_RD_to_use = M_RD

    md_u_flat = M_MD_u.flatten()
    rd_flat = M_RD_to_use.flatten() # Use the (potentially resized) M_RD

    valid_mask = (
        (~np.isnan(md_u_flat)) &
        (~np.isnan(rd_flat)) &
        (rd_flat > min_rd_thresh) & (rd_flat < (1.0 - min_rd_thresh)) &
        (md_u_flat > 1e-3) # M_MD_u values should also be sensible
    )
    X = md_u_flat[valid_mask].reshape(-1, 1)
    y = rd_flat[valid_mask].reshape(-1, 1)

    if X.shape[0] < min_points_for_fit:
        # print(f"Không đủ điểm hợp lệ ({X.shape[0]}) để thực hiện linear regression. Cần ít nhất {min_points_for_fit}.")
        return None, np.nan, np.nan

    try:
        reg = LinearRegression().fit(X, y)
        a = reg.coef_[0][0]
        b = reg.intercept_[0]
    except Exception as e:
        # print(f"Lỗi khi thực hiện Linear Regression: {e}")
        return None, np.nan, np.nan

    if abs(a) < 1e-7: # Avoid division by zero or very small slope
        # print(f"Hệ số góc 'a' quá nhỏ ({a}), có thể dẫn đến kết quả không ổn định. Không tinh chỉnh.")
        return None, a, b # Return current a,b but signal no refinement

    # Invert the transformation using the potentially resized M_RD_to_use
    M_MD_refined = np.full_like(M_RD_to_use, np.nan, dtype=np.float32) # Ensure float output
    
    # Use M_RD_to_use (which has the same shape as M_MD_u) for the refinement calculation
    m_rd_flat_for_pred = M_RD_to_use.flatten()
    valid_predict_mask = ~np.isnan(m_rd_flat_for_pred) # Use valid pixels from M_RD_to_use

    inverted_values = (m_rd_flat_for_pred[valid_predict_mask] - b) / a
    M_MD_refined.flat[valid_predict_mask] = inverted_values
    
    M_MD_refined[M_MD_refined < 1e-3] = np.nan
    M_MD_refined[np.isinf(M_MD_refined)] = np.nan # Handle potential infinities from division

    return M_MD_refined, a, b


def metric_to_relative(metric_depth_map):
    if metric_depth_map is None or np.all(np.isnan(metric_depth_map)):
        return np.full_like(metric_depth_map, 0.5, dtype=np.float32) if metric_depth_map is not None else None
    
    # Ensure input is float for percentile calculation if it's not already
    metric_depth_map_float = metric_depth_map.astype(np.float32)
    
    valid_depths = metric_depth_map_float[~np.isnan(metric_depth_map_float) & (metric_depth_map_float > 1e-3)]
    if valid_depths.size == 0:
        return np.full_like(metric_depth_map_float, 0.5, dtype=np.float32)
    
    min_d = np.percentile(valid_depths, 1)
    max_d = np.percentile(valid_depths, 99)
    
    if max_d <= min_d: # Handles cases where all valid depths are very close or identical
        # If max_d and min_d are equal, all valid points are the same depth.
        # Relative map could be all 0, all 1, or all 0.5. 0.5 is a neutral choice.
        return np.full_like(metric_depth_map_float, 0.5, dtype=np.float32)
        
    relative_map = (metric_depth_map_float - min_d) / (max_d - min_d)
    relative_map = np.clip(relative_map, 0, 1)
    relative_map[np.isnan(metric_depth_map_float)] = np.nan # Preserve NaNs from original metric map
    return relative_map

def evaluate_metric_depths(m_md_u, m_md_refined):
    if m_md_u is None or m_md_refined is None: return None
    if m_md_u.shape != m_md_refined.shape: return None # Should not happen if processed correctly
    
    valid_mask = (~np.isnan(m_md_u)) & (~np.isnan(m_md_refined)) & \
                 (m_md_u > 1e-3) & (m_md_refined > 1e-3)
    
    if not np.any(valid_mask): return None # No comparable valid pixels
    
    m_md_u_valid = m_md_u[valid_mask]
    m_md_refined_valid = m_md_refined[valid_mask]

    # Avoid division by zero if m_md_refined_valid can be zero (though >1e-3 filter should prevent)
    # Ensure m_md_refined_valid is not zero before division
    # The 1e-3 filter should make m_md_refined_valid always positive here
    
    relative_diff = np.abs(m_md_u_valid - m_md_refined_valid) / m_md_refined_valid * 100
    mean_diff = np.mean(relative_diff)
    return mean_diff

def compute_depth_metrics(pred, gt):
    if pred is None or gt is None: return None
    if pred.shape != gt.shape:
        # print(f"Shape mismatch in compute_depth_metrics. Pred: {pred.shape}, GT: {gt.shape}. Resizing GT.")
        gt_resized = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_LINEAR)
        if gt_resized.shape != pred.shape:
            # print("Resize failed in compute_depth_metrics. Returning None for metrics.")
            return {'abs_rel': np.nan, 'rmse': np.nan, 'delta1': np.nan, 'delta2': np.nan, 'delta3': np.nan}
        gt_to_use = gt_resized
    else:
        gt_to_use = gt
    
    # Ensure inputs are float for calculations
    pred_float = pred.astype(np.float32)
    gt_float = gt_to_use.astype(np.float32)

    mask = (~np.isnan(pred_float)) & (~np.isnan(gt_float)) & \
           (pred_float > 1e-3) & (gt_float > 1e-3) # Use 1e-3 for GT as well

    default_metrics = {'abs_rel': np.nan, 'rmse': np.nan, 'delta1': np.nan, 'delta2': np.nan, 'delta3': np.nan}
    if not np.any(mask): return default_metrics
    
    pred_valid = pred_float[mask]
    gt_valid = gt_float[mask]
    
    if gt_valid.size == 0: return default_metrics # Should be caught by np.any(mask)

    # Absolute Relative Error
    abs_rel = np.mean(np.abs(pred_valid - gt_valid) / gt_valid)
    # Root Mean Square Error
    rmse = np.sqrt(np.mean(np.square(pred_valid - gt_valid)))
    # Accuracy metrics (δ1, δ2, δ3)
    thresh = np.maximum((gt_valid / pred_valid), (pred_valid / gt_valid))
    delta1 = np.mean(thresh < 1.25).astype(np.float32)
    delta2 = np.mean(thresh < 1.25**2).astype(np.float32)
    delta3 = np.mean(thresh < 1.25**3).astype(np.float32)
    
    return {'abs_rel': abs_rel, 'rmse': rmse, 'delta1': delta1, 'delta2': delta2, 'delta3': delta3}

def process_single_image(rgb_path, depth_gt_path, unik3d_model, device):
    try:
        relative_depth_gt_original_size = load_direct_relative_depth_gt(depth_gt_path)
        if relative_depth_gt_original_size is None: return None

        m_md_u, m_md_u_shape_hw = create_unik3d_metric_depth(rgb_path, unik3d_model, device, target_height=480)
        if m_md_u is None: return None

        # Resize GT to match UniK3D's output shape *before* potential nan_to_num
        # This ensures that interpolation happens on the original data as much as possible.
        relative_depth_gt_resized = cv2.resize(relative_depth_gt_original_size,
                                             (m_md_u_shape_hw[1], m_md_u_shape_hw[0]),
                                             interpolation=cv2.INTER_LINEAR)
        
        # It's generally better to keep NaNs if possible and let downstream functions handle them,
        # unless a function specifically requires no NaNs.
        # `fit_linear_and_refine_metric_depth` handles NaNs in its inputs.
        # If `relative_depth_gt_resized` has NaNs from resizing or original, they'll be part of the `valid_mask`.
        # For this version, let's keep the nan_to_num as it was in the user's code,
        # assuming there's a reason for it (e.g., some metrics might not handle NaN well if not masked).
        # However, the `compute_depth_metrics` already handles NaNs.
        # Consider removing this nan_to_num if masking is sufficient.
        # For now, keeping it to match user's previous logic.
        # if np.any(np.isnan(relative_depth_gt_resized)):
        #    relative_depth_gt_resized = np.nan_to_num(relative_depth_gt_resized, nan=0.0) # Or use another fill value if 0.0 is problematic

        m_md_refined, fit_a, fit_b = fit_linear_and_refine_metric_depth(m_md_u, relative_depth_gt_resized)
        
        if m_md_refined is None:
            # print(f"Metric depth refinement failed for {os.path.basename(rgb_path)}.")
            # Optionally, one could still return raw metrics if m_md_u is valid and a suitable GT exists.
            # For this task, if refinement (the core step) fails, we consider the processing for this image incomplete.
            return None

        # Metrics for m_md_u (raw UniK3D) vs m_md_refined (refined output treated as pseudo-GT for this comparison)
        # This indicates how much the linear fit altered the UniK3D output.
        metrics_unik3d_vs_refined = compute_depth_metrics(m_md_u, m_md_refined)
        
        diff_percentage_unik3d_vs_refined = evaluate_metric_depths(m_md_u, m_md_refined)

        # Convert the refined METRIC depth (m_md_refined) to RELATIVE depth
        relative_depth_refined_from_metric = metric_to_relative(m_md_refined)
        
        # Compare this new relative depth with the (resized) ground truth RELATIVE depth
        metrics_relative_refined_vs_relative_gt = compute_depth_metrics(
            relative_depth_refined_from_metric, 
            relative_depth_gt_resized # Use the same GT that was used for fitting
        )
        
        # Include m_md_refined in results if needed (it's a large array)
        # For JSON, usually we store metrics, not the full depth map, unless specified.
        # The request implies storing information from process_single_image(), which includes m_md_refined.

        return {
            'image_id': os.path.splitext(os.path.basename(rgb_path))[0],
            'linear_fit_slope_a': fit_a,
            'linear_fit_intercept_b': fit_b,
            'diff_percentage_unik3d_vs_refined': diff_percentage_unik3d_vs_refined,
            'metrics_unik3d_vs_refined': metrics_unik3d_vs_refined,
            'metrics_relative_refined_vs_relative_gt': metrics_relative_refined_vs_relative_gt,
            'm_md_refined_shape': m_md_refined.shape, # Store shape instead of full array for smaller JSON
            'm_md_refined_min': np.nanmin(m_md_refined) if m_md_refined is not None else None,
            'm_md_refined_max': np.nanmax(m_md_refined) if m_md_refined is not None else None,
            'm_md_refined_mean': np.nanmean(m_md_refined) if m_md_refined is not None else None,
            # 'm_md_refined': m_md_refined # Uncomment if full array is truly needed in JSON
        }

    except FileNotFoundError as e:
        # print(f"Lỗi file không tìm thấy trong process_single_image: {e}")
        return None
    except ValueError as e:
        # print(f"Lỗi giá trị trong process_single_image cho {os.path.basename(rgb_path)}: {e}")
        return None
    except Exception as e:
        # print(f"Lỗi không xác định trong process_single_image cho {os.path.basename(rgb_path)}: {e}")
        # import traceback
        # traceback.print_exc() # For debugging detailed errors
        return None

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, o):
        if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(o)
        elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
            if np.isnan(o):
                return None  # Represent NaN as null in JSON
            if np.isinf(o):
                # Represent Inf as a large number string, or None, or specific string
                # For simplicity, None is often best for JSON compatibility.
                return None # Represent Inf as null
            return float(o)
        elif isinstance(o, np.bool_):
            return bool(o)
        elif isinstance(o, np.ndarray):
            if o.ndim == 0: # Handle 0-d arrays (scalars)
                return self.default(o.item()) # Recursively call default for the item
            
            # For floating point arrays, handle NaN/Inf carefully BEFORE converting to list
            if np.issubdtype(o.dtype, np.floating):
                # Create a copy of the array with dtype=object to allow storing None
                obj_array = o.astype(object)
                
                # Create masks from the original float array 'o'
                nan_mask = np.isnan(o)
                inf_mask = np.isinf(o) # Catches both +inf and -inf
                
                # Replace NaN and Inf values with None in the object array
                obj_array[nan_mask] = None
                obj_array[inf_mask] = None
                
                return obj_array.tolist()
            else:
                # For non-float arrays (int, bool, or already object if not float),
                # tolist() is generally safe. If it's an object array that might
                # contain np.nan/np.inf floats, the recursive call to self.default
                # for individual elements (when json.dump processes the list items)
                # will handle them via the scalar np.float_ check.
                return o.tolist()
        return super(NumpyEncoder, self).default(o)


def generate_json_outputs(base_dataset_dir, json_output_dir_base):
    print("--- Bắt đầu quy trình tạo file JSON ---")

    if not UNIK3D_AVAILABLE:
        print("UniK3D không khả dụng. Dừng chương trình.")
        return

    try:
        unik3d_model, device = instantiate_unik3d_model()
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi khởi tạo model UniK3D: {e}. Dừng chương trình.")
        return

    images_input_dir = os.path.join(base_dataset_dir, "images")
    depths_input_dir = os.path.join(base_dataset_dir, "depths")
    
    json_output_dir = os.path.join(json_output_dir_base, "metric_depth_pipeline_outputs") 

    if not os.path.isdir(images_input_dir):
        print(f"Thư mục ảnh RGB không tìm thấy: {images_input_dir}. Dừng chương trình.")
        return
    if not os.path.isdir(depths_input_dir):
        print(f"Thư mục depths GT không tìm thấy: {depths_input_dir}. Dừng chương trình.")
        return

    os.makedirs(json_output_dir, exist_ok=True)
    print(f"Các file JSON sẽ được lưu vào: {json_output_dir}")

    image_filenames = sorted([f for f in os.listdir(images_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not image_filenames:
        print(f"Không tìm thấy file ảnh nào trong {images_input_dir}. Dừng chương trình.")
        return

    processed_count = 0
    error_skip_count = 0
    total_images = len(image_filenames)
    print(f"Tìm thấy {total_images} ảnh RGB để xử lý.")

    for i, image_filename in enumerate(image_filenames):
        image_id = os.path.splitext(image_filename)[0]
        
        rgb_path = os.path.join(images_input_dir, image_filename)
        depth_gt_filename = f"{image_id}_depth.png" # Assuming depth is always .png
        depth_gt_path = os.path.join(depths_input_dir, depth_gt_filename)

        if not os.path.exists(depth_gt_path):
            # print(f"  Cảnh báo: File depth GT không tìm thấy: {depth_gt_path} cho ảnh {image_filename}. Bỏ qua.")
            error_skip_count += 1
            continue
        
        results_dict = process_single_image(rgb_path, depth_gt_path, unik3d_model, device)

        if results_dict is not None:
            json_filename = f"{image_id}.json"
            json_filepath = os.path.join(json_output_dir, json_filename)
            
            try:
                # Remove the large array if it exists and only summary stats were intended
                if 'm_md_refined' in results_dict and isinstance(results_dict['m_md_refined'], np.ndarray):
                    del results_dict['m_md_refined'] # Remove if only stats are needed

                with open(json_filepath, 'w') as f:
                    json.dump(results_dict, f, cls=NumpyEncoder, indent=4)
                processed_count += 1
            except Exception as e:
                print(f"  Lỗi khi lưu file JSON {json_filepath}: {e}")
                error_skip_count += 1
        else:
            # print(f"  Không có kết quả (hoặc lỗi) từ process_single_image cho {image_filename}. Bỏ qua.")
            error_skip_count += 1
        
        if (i + 1) % 20 == 0 or (i + 1) == total_images:
             print(f"Tiến độ: {i+1}/{total_images} | Thành công: {processed_count} | Lỗi/Bỏ qua: {error_skip_count}")

    print(f"\n--- Hoàn tất quy trình tạo file JSON ---")
    print(f"Tổng số ảnh RGB đầu vào được quét: {total_images}")
    print(f"Số file JSON đã tạo thành công: {processed_count}")
    print(f"Số trường hợp lỗi hoặc bị bỏ qua (thiếu depth, lỗi xử lý): {error_skip_count}")

def convert_relative_to_metric_depth_from_json(json_path,
                                               relative_depth_image_path,
                                               load_gt_func=load_direct_relative_depth_gt):
    """
    Converts a relative depth image to a metric depth ndarray using transformation
    parameters (slope 'a' and intercept 'b') stored in a JSON file.

    Args:
        json_path (str): Path to the JSON file containing 'linear_fit_slope_a',
                         'linear_fit_intercept_b', and 'm_md_refined_shape'.
        relative_depth_image_path (str): Path to the relative depth image (e.g., a PNG)
                                         to be converted.
        load_gt_func (function): Function to load and preprocess the relative depth image.
                                 Defaults to `load_direct_relative_depth_gt`.

    Returns:
        numpy.ndarray: The converted metric depth map (float32), or None if conversion fails.
                       Values < 1e-3 or resulting from invalid operations (like division by
                       near-zero slope) will be NaN.
    """
    if not os.path.exists(json_path):
        print(f"Error: JSON file not found: {json_path}")
        return None
    if not os.path.exists(relative_depth_image_path):
        print(f"Error: Relative depth image not found: {relative_depth_image_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            params = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON file {json_path}: {e}")
        return None
    except Exception as e:
        print(f"Error: Could not read JSON file {json_path}: {e}")
        return None

    slope_a = params.get('linear_fit_slope_a')
    intercept_b = params.get('linear_fit_intercept_b')
    # m_md_refined_shape is typically [height, width]
    target_shape_hw = params.get('m_md_refined_shape')

    if slope_a is None or intercept_b is None:
        print(f"Error: JSON file {json_path} is missing 'linear_fit_slope_a' or 'linear_fit_intercept_b'.")
        return None
    if target_shape_hw is None:
        print(f"Error: JSON file {json_path} is missing 'm_md_refined_shape'.")
        return None
    if not isinstance(target_shape_hw, list) or len(target_shape_hw) != 2:
        print(f"Error: 'm_md_refined_shape' in {json_path} is not a list of 2 elements [height, width]. Found: {target_shape_hw}")
        return None
    
    target_shape_hw_tuple = tuple(target_shape_hw) # For functions expecting a tuple

    if abs(slope_a) < 1e-7: # Avoid division by zero or very small slope
        print(f"Error: Slope 'a' from JSON ({slope_a}) is too small for reliable metric depth conversion.")
        # Return a map full of NaNs with the target shape
        return np.full(target_shape_hw_tuple, np.nan, dtype=np.float32)

    try:
        # Load the new relative depth map, and critically, resize it to the shape
        # that m_md_refined had (which is the shape for which 'a' and 'b' are valid).
        new_relative_depth_map = load_gt_func(relative_depth_image_path,
                                              target_shape_hw=target_shape_hw_tuple)
        if new_relative_depth_map is None:
            # load_gt_func should raise an error if it fails, but double-check
            print(f"Error: Loading relative depth map from {relative_depth_image_path} failed.")
            return None
        
        # Ensure it's float32 for calculations
        new_relative_depth_map = new_relative_depth_map.astype(np.float32)

    except FileNotFoundError:
        # This is already checked above, but good to have specific error here if load_gt_func raises it.
        print(f"Error: Relative depth image not found by load function: {relative_depth_image_path}")
        return None
    except Exception as e:
        print(f"Error processing relative depth image {relative_depth_image_path}: {e}")
        return None

    # Apply the inverse transformation: MD = (RD - b) / a
    metric_depth_map = (new_relative_depth_map - intercept_b) / slope_a

    # Post-processing similar to fit_linear_and_refine_metric_depth
    metric_depth_map[metric_depth_map < 1e-3] = np.nan
    metric_depth_map[np.isinf(metric_depth_map)] = np.nan # Handle potential infinities

    return metric_depth_map

if __name__ == "__main__":
    # --- Configuration ---
    # Adjust these paths as per your environment
    BASE_DATASET_DIR = "/root/PhysicalAI_Dataset/test"
    # This will be the parent directory for the specific JSON output folder
    MAIN_OUTPUT_DIR_BASE = "/root/PhysicalAI_Dataset/predict_results" 
    
    # If the error message path `/root/PhysicalAI_Dataset/predict_results/` is intended, then:
    # MAIN_OUTPUT_DIR_BASE = "/root/PhysicalAI_Dataset/predict_results/"

    generate_json_outputs(
        base_dataset_dir=BASE_DATASET_DIR,
        json_output_dir_base=MAIN_OUTPUT_DIR_BASE
    )