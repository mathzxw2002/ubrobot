import gradio as gr
import random
import time

from pathlib import Path

import qwen_vl_utils
import transformers

ROOT = Path(__file__).parents[2]
SEPARATOR = "-" * 20


#CosmosReason1Infer cosmos_infer

'''with gr.Blocks() as demo:

    # Load model
    model_name = "/home/sany/.cache/modelscope//hub/models/nv-community/Cosmos-Reason1-7B"
    model = transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="float16", device_map="auto"
    )
    processor: transformers.Qwen2_5_VLProcessor = (
        transformers.AutoProcessor.from_pretrained(model_name)
    )

    # Create inputs
    conversation = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"./sample.mp4",
                    "fps": 4,
                    # 6422528 = 8192 * 28**2 = vision_tokens * (2*spatial_patch_size)^2
                    "total_pixels": 6422528,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    # Process inputs
    text = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    # Run inference
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(SEPARATOR)
    print(output_text[0])
    print(SEPARATOR)


    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        bot_message = output_text[0] #random.choice(["How are you?", "Today is a great day", "I'm very hungry"])
        chat_history.append({"role": "user", "content": message})
        chat_history.append({"role": "assistant", "content": bot_message})
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(server_name="0.0.0.0", server_port=7861)
'''

def create_chatbot_interface() -> gr.Blocks:
    """
    创建聊天机器人界面的独立函数
    :return: 配置好的 Gradio Blocks 实例
    """
    # 单独封装 Blocks 界面配置
    with gr.Blocks(title="机器人") as demo:
        # 聊天机器人组件（设置高度和标签）
        chatbot = gr.Chatbot(
            label="聊天对话",
            height=500,
            bubble_full_width=False,  # 气泡不占满宽度
            show_copy_button=True      # 显示复制按钮
        )
        
        # 文本输入框（支持回车提交，设置占位提示）
        msg = gr.Textbox(
            label="输入消息",
            placeholder="请输入您的消息，按回车发送... 点击输入框可自动填充示例消息",
            lines=1,
            max_lines=3  # 最多允许3行输入
        )
        
        # 清除按钮（清空输入框和聊天记录）
        clear = gr.ClearButton(
            components=[msg, chatbot],
            label="清空对话",
            variant="secondary"  # 次要按钮样式
        )

        def respond(message: str, chat_history: list) -> tuple[str, list]:
            """
            聊天响应函数
            :param message: 用户输入的消息
            :param chat_history: 历史聊天记录列表
            :return: 清空的输入框内容 + 更新后的聊天记录
            """
            # 如果用户输入为空，直接返回
            if not message.strip():
                return "", chat_history
            
            # 原样返回用户输入（回声功能）
            # 如需随机回复，可启用下面的注释代码：
            # bot_responses = ["How are you?", "Today is a great day", "I'm very hungry", "Nice to meet you!"]
            # bot_message = random.choice(bot_responses)
            bot_message = f"回声：{message}"  # 增加回声标识，更清晰
            
            # 更新聊天记录（Gradio 3.x+ 推荐使用列表格式，每个元素是(用户消息, 机器人消息)元组）
            chat_history.append((message, bot_message))
            
            # 模拟思考时间（2秒），让交互更自然
            time.sleep(2)
            
            # 返回：清空输入框 + 更新后的聊天记录
            return "", chat_history

        # 文本框点击事件响应函数：点击时自动填充随机示例消息
        def on_textbox_click() -> str:
            """
            文本框点击事件响应函数：点击时自动填充随机示例消息
            :return: 要填充到文本框的内容
            """
            example_messages = [
                "你好！这个聊天机器人怎么用？",
                "今天天气真好啊～",
                "能介绍一下你的功能吗？",
                "echo test: 测试回声功能",
                "Gradio界面真方便！"
            ]
            return random.choice(example_messages)

        # 绑定事件：输入框按回车提交
        msg.submit(
            fn=respond,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot]
        )
        
        # 添加点击发送按钮（适合移动设备）
        with gr.Row():
            submit_btn = gr.Button("发送", variant="primary")
            submit_btn.click(
                fn=respond,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
        
        # 绑定文本框点击事件
        msg.click(
            fn=on_textbox_click,
            inputs=[],
            outputs=[msg]
        )
    
    return demo  # 返回配置好的 Blocks 实例

# 主函数：程序入口
if __name__ == "__main__":
    # 调用函数创建界面实例
    demo = create_chatbot_interface()
    
    # 启动应用
    demo.launch(
        server_name="0.0.0.0",  # 允许局域网访问
        server_port=7862,       # 端口号
        share=False,            # 是否生成公共链接（True时会生成临时公共URL）
        inbrowser=True,         # 自动打开浏览器
        show_error=True         # 显示错误信息
    )