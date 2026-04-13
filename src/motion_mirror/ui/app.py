def create_app():
    import gradio as gr

    with gr.Blocks() as demo:
        gr.Markdown("# Motion Mirror")
        gr.Markdown("Initial scaffold for the local-first motion transfer UI.")
    return demo
