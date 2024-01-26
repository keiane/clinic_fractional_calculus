import gradio as gr

# CSS theme styling
theme = gr.themes.Base(
    font=[gr.themes.GoogleFont('Montserrat'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    primary_hue="emerald",
    secondary_hue="emerald",
    neutral_hue="zinc"
).set(
    body_text_color='*neutral_950',
    body_text_color_subdued='*neutral_950',
    block_shadow='*shadow_drop_lg',
    button_shadow='*shadow_drop_lg',
    block_title_text_color='*neutral_950',
    block_title_text_weight='500',
    slider_color='*secondary_600'
)