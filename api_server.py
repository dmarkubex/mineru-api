from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import shutil
import json
import time
import mimetypes
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 启用GPU支持
os.environ["FLAGS_use_gpu"] = "1"
os.environ["PADDLEPADDLE_INSTALL_GPU"] = "1"
logger.info("已启用GPU支持")

# 修复环境变量，确保能找到CUDA和Paddle库
paddle_lib_path = "/opt/mineru_venv/lib/python3.10/site-packages/paddle/libs"
if os.path.exists(paddle_lib_path):
    os.environ['LD_LIBRARY_PATH'] = f"{paddle_lib_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    logger.info(f"已设置LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')}")

# 导入 MinerU 相关库
try:
    from magic_pdf.data.data_reader_writer import FileBasedDataWriter
    from magic_pdf.data.dataset import PymuDocDataset
    from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
    from magic_pdf.config.enums import SupportedPdfParseMethod
    from magic_pdf.data.read_api import read_local_office, read_local_images
    logger.info("成功导入 magic_pdf 库")
except ImportError as e:
    logger.error(f"导入 magic_pdf 库时出错: {str(e)}")
    raise

app = FastAPI(title="MinerU API", description="MinerU 文档处理 API 服务")

# 数据存储路径
DATA_DIR = "/app/data"
UPLOADS_DIR = f"{DATA_DIR}/uploads"
RESULTS_DIR = f"{DATA_DIR}/results"

# 确保目录存在
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 存储任务状态
tasks = {}

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    message: Optional[str] = None

class TaskResultResponse(BaseModel):
    markdown: Optional[str] = None
    content_list: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[str]] = None
    processing_time: Optional[float] = None

@app.post("/convert", response_model=TaskResponse)
async def convert_document(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    异步转换上传的文档（PDF、MS Office、图像）
    """
    # 生成任务ID
    task_id = str(uuid.uuid4())
    
    # 创建任务目录
    task_upload_dir = os.path.join(UPLOADS_DIR, task_id)
    task_result_dir = os.path.join(RESULTS_DIR, task_id)
    task_image_dir = os.path.join(task_result_dir, "images")
    
    os.makedirs(task_upload_dir, exist_ok=True)
    os.makedirs(task_result_dir, exist_ok=True)
    os.makedirs(task_image_dir, exist_ok=True)
    
    # 保存上传的文件
    file_path = os.path.join(task_upload_dir, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # 初始化任务状态
    tasks[task_id] = {
        "status": "queued",
        "progress": 0,
        "start_time": time.time(),
        "file_name": file.filename
    }
    
    # 添加后台任务
    background_tasks.add_task(
        process_document, 
        task_id=task_id,
        file_path=file_path,
        result_dir=task_result_dir,
        image_dir=task_image_dir
    )
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "文档已成功上传并加入处理队列"
    }

@app.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    获取任务处理状态
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    
    return {
        "task_id": task_id,
        "status": tasks[task_id]["status"],
        "progress": tasks[task_id].get("progress"),
        "message": tasks[task_id].get("message")
    }

@app.get("/result/{task_id}", response_model=TaskResultResponse)
async def get_task_result(task_id: str):
    """
    获取任务处理结果
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    
    if tasks[task_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"任务 {task_id} 尚未完成，当前状态: {tasks[task_id]['status']}")
    
    task_result_dir = os.path.join(RESULTS_DIR, task_id)
    logger.info(f"获取任务 {task_id} 的结果，结果目录: {task_result_dir}")
    
    # 获取结果文件
    file_name = tasks[task_id]["file_name"]
    name_without_suff = os.path.splitext(file_name)[0]
    
    md_path = os.path.join(task_result_dir, f"{name_without_suff}.md")
    content_list_path = os.path.join(task_result_dir, f"{name_without_suff}_content_list.json")
    
    logger.info(f"查找Markdown文件: {md_path}")
    logger.info(f"查找内容列表文件: {content_list_path}")
    
    result = {}
    
    if os.path.exists(md_path):
        logger.info(f"找到Markdown文件")
        with open(md_path, "r", encoding='utf-8') as f:
            result["markdown"] = f.read()
    else:
        logger.warning(f"未找到Markdown文件: {md_path}")
    
    if os.path.exists(content_list_path):
        logger.info(f"找到内容列表文件")
        with open(content_list_path, "r", encoding='utf-8') as f:
            result["content_list"] = json.load(f)
    else:
        logger.warning(f"未找到内容列表文件: {content_list_path}")
    
    # 获取图像列表
    image_dir = os.path.join(task_result_dir, "images")
    if os.path.exists(image_dir):
        images = os.listdir(image_dir)
        result["images"] = [f"/image/{task_id}/{image}" for image in images]
        logger.info(f"找到 {len(images)} 个图像文件")
    
    # 获取处理统计信息
    result["processing_time"] = tasks[task_id].get("processing_time")
    
    # 如果结果中没有任何内容，记录错误
    if not result:
        logger.error(f"任务 {task_id} 没有找到任何结果文件")
        raise HTTPException(status_code=500, detail="未找到任何处理结果")
    
    return result

@app.get("/image/{task_id}/{image_name}")
async def get_image(task_id: str, image_name: str):
    """
    获取任务处理结果中的图像
    """
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    
    image_path = os.path.join(RESULTS_DIR, task_id, "images", image_name)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail=f"图像 {image_name} 不存在")
    
    return FileResponse(image_path)

async def process_document(task_id: str, file_path: str, result_dir: str, image_dir: str):
    """
    后台处理文档任务
    """
    try:
        logger.info(f"开始处理任务 {task_id}")
        logger.info(f"文件路径: {file_path}")
        logger.info(f"结果目录: {result_dir}")
        logger.info(f"图片目录: {image_dir}")
        
        # 更新任务状态
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 0.1
        tasks[task_id]["start_time"] = time.time()
        tasks[task_id]["file_name"] = os.path.basename(file_path)
        tasks[task_id]["message"] = "正在读取文件..."
        
        # 准备输出目录
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        
        # 设置图片相对路径（用于Markdown中的图片引用）
        image_dir_name = "images"
        
        # 创建写入器
        image_writer = FileBasedDataWriter(image_dir)
        result_writer = FileBasedDataWriter(result_dir)
        
        # 获取文件名（不含扩展名）
        name_without_suff = os.path.splitext(os.path.basename(file_path))[0]
        logger.info(f"处理文件: {name_without_suff}")
        
        # 读取文件
        with open(file_path, "rb") as f:
            file_bytes = f.read()
            logger.info(f"成功读取文件 {file_path}, 大小: {len(file_bytes)} bytes")
            
        # 检测文件类型并处理
        try:
            if file_bytes.startswith(b'\xFF\xD8\xFF') or file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):  # 图片文件
                logger.info("检测到图片文件，开始处理...")
                tasks[task_id]["message"] = "正在处理图片..."
                
                # 创建临时文件
                ext = ".jpg" if file_bytes.startswith(b'\xFF\xD8\xFF') else ".png"
                temp_path = os.path.join(result_dir, f"temp{ext}")
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)
                logger.info(f"已创建临时文件: {temp_path}")
                
                try:
                    # 处理图片
                    logger.info("开始读取图片...")
                    ds = read_local_images(temp_path)[0]
                    logger.info("图片读取成功，开始分析...")
                    
                    # 直接使用标准处理流程
                    pipe_result = ds.apply(doc_analyze, ocr=True).pipe_ocr_mode(image_writer)
                    logger.info("图片处理完成")
                    
                    # 保存Markdown结果
                    logger.info("保存Markdown结果")
                    md_file = f"{name_without_suff}.md"
                    pipe_result.dump_md(result_writer, md_file, image_dir_name)
                    logger.info(f"Markdown文件已保存: {md_file}")
                    
                    # 读取生成的Markdown内容
                    md_path = os.path.join(result_dir, md_file)
                    with open(md_path, "r", encoding='utf-8') as f:
                        md_content = f.read()
                    logger.info(f"Markdown内容长度: {len(md_content)}")
                    
                    # 获取内容列表（如果需要）
                    content_list = pipe_result.get_content_list(image_dir_name)
                    
                    # 获取图片列表
                    images = []
                    if os.path.exists(image_dir):
                        images = [f"/image/{task_id}/{img}" for img in os.listdir(image_dir)]
                        logger.info(f"找到 {len(images)} 个图像文件")
                    
                    # 计算处理时间
                    processing_time = time.time() - tasks[task_id]["start_time"]
                    
                    # 更新任务状态
                    tasks[task_id]["status"] = "completed"
                    tasks[task_id]["progress"] = 1.0
                    tasks[task_id]["message"] = "处理完成"
                    tasks[task_id]["processing_time"] = processing_time
                    tasks[task_id]["result"] = {
                        "markdown": md_content,
                        "content_list": content_list,
                        "images": images
                    }
                    
                    logger.info(f"任务 {task_id} 处理完成，耗时: {processing_time:.2f}秒")
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.info(f"已清理临时文件: {temp_path}")
            
            elif file_bytes.startswith(b'%PDF'):
                logger.info("检测到PDF文件，开始处理...")
                tasks[task_id]["message"] = "正在处理 PDF 文档..."
                
                # 创建数据集实例
                ds = PymuDocDataset(file_bytes)
                logger.info("成功创建PymuDocDataset实例")
                
                # 分类文档类型
                doc_type = ds.classify()
                logger.info(f"文档类型分类结果: {doc_type}")
                
                # 推理
                if doc_type == SupportedPdfParseMethod.OCR:
                    logger.info("使用 OCR 模式处理文档")
                    tasks[task_id]["message"] = "正在使用 OCR 模式分析文档..."
                    infer_result = ds.apply(doc_analyze, ocr=True)
                    logger.info("OCR模式推理完成")
                    pipe_result = infer_result.pipe_ocr_mode(image_writer)
                    logger.info("OCR模式管道处理完成")
                else:
                    logger.info("使用文本模式处理文档")
                    tasks[task_id]["message"] = "正在使用文本模式分析文档..."
                    infer_result = ds.apply(doc_analyze, ocr=False)
                    logger.info("文本模式推理完成")
                    pipe_result = infer_result.pipe_txt_mode(image_writer)
                    logger.info("文本模式管道处理完成")
                
                # 保存处理结果
                logger.info("开始保存处理结果")
                tasks[task_id]["progress"] = 0.8
                tasks[task_id]["message"] = "正在保存结果..."
                
                # 获取并保存 Markdown 内容
                logger.info("生成Markdown内容")
                md_content = pipe_result.get_markdown("images")
                logger.info(f"Markdown内容长度: {len(md_content) if md_content else 0}")
                md_file = f"{name_without_suff}.md"
                pipe_result.dump_md(result_writer, md_file, "images")
                logger.info(f"Markdown文件已保存: {md_file}")
                
                # 获取并保存内容列表
                logger.info("生成内容列表")
                content_list = pipe_result.get_content_list("images")
                content_list_file = f"{name_without_suff}_content_list.json"
                pipe_result.dump_content_list(result_writer, content_list_file, "images")
                logger.info(f"内容列表已保存: {content_list_file}")
                
                # 获取并保存中间 JSON
                logger.info("生成中间JSON")
                middle_json = pipe_result.get_middle_json()
                middle_json_file = f"{name_without_suff}_middle.json"
                pipe_result.dump_middle_json(result_writer, middle_json_file)
                logger.info(f"中间JSON已保存: {middle_json_file}")
                
                # 检查生成的文件
                logger.info("检查生成的文件:")
                for file in os.listdir(result_dir):
                    logger.info(f"- {file}")
                
                # 计算处理时间
                processing_time = time.time() - tasks[task_id]["start_time"]
                
                # 更新任务状态
                tasks[task_id]["status"] = "completed"
                tasks[task_id]["progress"] = 1.0
                tasks[task_id]["message"] = "处理完成"
                tasks[task_id]["processing_time"] = processing_time
                tasks[task_id]["result"] = {
                    "markdown": md_content,
                    "content_list": content_list,
                    "middle_json": middle_json
                }
                
                logger.info(f"任务 {task_id} 处理完成，耗时: {processing_time:.2f}秒")
                
            elif file_bytes[0:4] in [b'PK\x03\x04', b'PK\x05\x06', b'PK\x07\x08']:  # Office 文件
                logger.info("检测到Office文件，开始处理...")
                tasks[task_id]["message"] = "正在处理 Office 文档..."
                
                # 创建临时文件
                ext = os.path.splitext(file_path)[1]
                temp_path = os.path.join(result_dir, f"temp{ext}")
                with open(temp_path, "wb") as f:
                    f.write(file_bytes)
                logger.info(f"已创建临时文件: {temp_path}")
                
                try:
                    # 按照指定逻辑处理Office文件
                    logger.info("开始按照指定逻辑处理Office文件...")
                    
                    # prepare env - 严格按照提供的示例
                    local_image_dir, local_md_dir = os.path.join(result_dir, "images"), result_dir
                    image_dir = str(os.path.basename(local_image_dir))
                    
                    os.makedirs(local_image_dir, exist_ok=True)
                    
                    image_writer = FileBasedDataWriter(local_image_dir)
                    md_writer = FileBasedDataWriter(local_md_dir)
                    
                    # proc
                    input_file = temp_path
                    input_file_name = os.path.basename(file_path).split(".")[0]
                    logger.info(f"处理文件: {input_file}, 输出文件名: {input_file_name}")
                    
                    ds = read_local_office(input_file)[0]
                    logger.info("Office文件读取成功，开始分析...")
                    
                    # 使用与示例代码完全相同的处理链
                    ds.apply(doc_analyze, ocr=True).pipe_txt_mode(image_writer).dump_md(
                        md_writer, f"{input_file_name}.md", image_dir
                    )
                    logger.info("Office文件处理完成")
                    
                    # 读取生成的Markdown内容
                    md_path = os.path.join(local_md_dir, f"{input_file_name}.md")
                    logger.info(f"尝试读取Markdown文件: {md_path}")
                    with open(md_path, "r", encoding='utf-8') as f:
                        md_content = f.read()
                    logger.info(f"Markdown内容长度: {len(md_content)}")
                    
                    # 获取图片列表
                    images = []
                    if os.path.exists(local_image_dir):
                        images = [f"/image/{task_id}/{img}" for img in os.listdir(local_image_dir)]
                        logger.info(f"找到 {len(images)} 个图像文件")
                    
                    # 计算处理时间
                    processing_time = time.time() - tasks[task_id]["start_time"]
                    
                    # 更新任务状态
                    tasks[task_id]["status"] = "completed"
                    tasks[task_id]["progress"] = 1.0
                    tasks[task_id]["message"] = "处理完成"
                    tasks[task_id]["processing_time"] = processing_time
                    tasks[task_id]["result"] = {
                        "markdown": md_content,
                        "content_list": None,  # 示例代码中没有获取content_list
                        "images": images
                    }
                    
                    logger.info(f"任务 {task_id} 处理完成，耗时: {processing_time:.2f}秒")
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.info(f"已清理临时文件: {temp_path}")
            
            else:
                raise ValueError("不支持的文件类型，请上传 PDF、Office 文档或图片文件")
            
        except Exception as e:
            logger.error(f"处理文件时出错: {str(e)}")
            tasks[task_id]["status"] = "failed"
            tasks[task_id]["message"] = f"处理失败: {str(e)}"
            raise ValueError(f"处理文件时出错: {str(e)}")
            
    except Exception as e:
        logger.error(f"任务 {task_id} 失败: {str(e)}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["message"] = f"处理失败: {str(e)}"
        
        # 记录错误日志
        error_log_path = os.path.join(result_dir, "error.log")
        with open(error_log_path, "w") as f:
            f.write(str(e))
    finally:
        # 清理上传文件
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"清理上传文件: {file_path}")

@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)