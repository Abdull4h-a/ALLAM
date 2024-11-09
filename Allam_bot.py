# -*- coding: utf-8 -*-
import logging
import pandas as pd
from telegram.helpers import escape_markdown
import torch
import torch.nn as nn
from transformers import AutoModel
from ibm_watsonx_ai.foundation_models import Model
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes

# Set logging level to WARNING to suppress INFO logs
logging.basicConfig(level=logging.WARNING)

# Bot configuration
TELEGRAM_BOT_TOKEN = "7789650517:AAHbOf2Fl6vT3Atx2Kp-LyfNUnmWKbTAcdM"
ALLAM_API_KEY = "JsA-9hFRsH1msnDzUkcRpoUhSi3r3dfQMAj8Ic9yRiNg"
ALLAM_PROJECT_ID = "43ada3b1-5d11-42e8-9ef1-c718b99c05e1"
ALLAM_MODEL_ID = "sdaia/allam-1-13b-instruct"

# Load data and mappings
try:
    test_df = pd.read_csv('train_data.csv')
    author_to_idx_mapping = {name: i for i, name in enumerate(test_df['AuthorName'].sort_values().unique().tolist())}
    idx_to_author_mapping = {v: k for k, v in author_to_idx_mapping.items()}
except FileNotFoundError:
    logging.error("train_data.csv not found. Ensure the dataset is available.")
    test_df = pd.DataFrame()
    author_to_idx_mapping = {}
    idx_to_author_mapping = {}

# Set up device and models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define classifier model
class TextClassifier(nn.Module):
    def __init__(self, input_size=1024, hidden_size=512, num_classes=len(author_to_idx_mapping)):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = torch.softmax(x, dim=-1)
        return x

# Initialize the embedding model
try:
    embedding_model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to(device)
except Exception as e:
    logging.error(f"Error loading embedding model: {e}")
    embedding_model = None

# Load the after-training model (fine-tuned model) on the CPU
classification_head_after = TextClassifier(input_size=1024, hidden_size=1024)
try:
    classification_head_after.load_state_dict(
        torch.load('author_classifier_head_jina_embeddings.pt', map_location=torch.device('cpu'))
    )
    classification_head_after.to(device)
    classification_head_after.eval()
except FileNotFoundError:
    logging.error("Fine-tuned model file not found.")

# Initialize Allam API for rewriting
allam_model = Model(
    model_id=ALLAM_MODEL_ID,
    params={"max_new_tokens": 500},
    credentials={"url": "https://eu-de.ml.cloud.ibm.com", "apikey": ALLAM_API_KEY},
    project_id=ALLAM_PROJECT_ID
)

def rewrite_text_with_allam(author_name, text):
    """
    Rewrites the input text in the style of the specified author using the ALLAM model and PROMPT_TEMPLATE.

    Args:
    - author_name (str): The name of the author whose style should be applied.
    - text (str): The original text to rewrite.

    Returns:
    - str: The rewritten text or an error message if the process fails.
    """
    # Define the PROMPT_TEMPLATE
    PROMPT_TEMPLATE = f"""
    1. أنت كاتب عربي خبير.
    2. سيتم تزويدك بنص إدخال ونص مرجعي.
    3. مهمتك هي إعادة صياغة النص المُدخل وإعادة كتابته ليتوافق مع أسلوب كتابة النص المرجعي.
    4. يمكنك تغيير المفردات والقواعد وما إلى ذلك لمطابقة نمط كتابة النص الناتج مع النص المرجعي، ولكن لا ينبغي أن يتغير المعنى الفعلي للنص المدخل.
    5. لا تخرج أي شيء إضافي. فقط النص المعاد كتابته. لا تقم بإخراج وصف النص.

    <Example_1>
        <مرجعي>
        أيهما أكبر: الكون أم الحياة الإنسانية؟ إن الحياة إن لم تكن لها غاية بعيدة موصولة بالغاية التي يسعى إليها الكون برمته فهي ولا ريب أصغر من أن تقاس إليه، أو يفاضل بينها وبينه. وقد كان يكفينا على هذا الفرض كرتنا الأرضية وحدها أو نظام واحد من أنظمة الشموس التي لا عداد لها. وإذا كانت الحياة الإنسانية هي الحس الشاعر المفرد في الوجود، فلِمَ لم يكن لها من الإحساس القدر الكافي لمعرفة الوجود حق المعرفة؟ ولِمَ لَمْ يتناسب العارف والمعروف أو يتقاربا؟ ألا نفهم من ذلك أنه لا بد في الوجود من قدرة تعرفه المعرفة الخليقة به؟ هذا هو الخاطر الذي قام بنفسي عند نظم الأبيات الآتية.
        </مرجعي>

        <إدخال>
        لِمَ قبُح الثناء في الوجه حتى تواطئوا على تزييفه؟ ولِمَ حسُن في المغيب حتى تُمُنِّيَ ذلك بكل معنًى؟ ألأن الثناء في الوجه أشبه الملق والخديعة وفي المغيب أشبه الإخلاص والتَّكْرِمَة أم لغير ذلك؟ قال أبو علي مسكويه، رحمه الله: لما كان الثناء في الوجه على الأكثر إعارة شهادةٍ بفضائل النفس وخديعة الإنسان بهذه الشهادة حتى صار ذلك — لاغتراره وتركه كثيرًا من الاجتهاد في تحصيل الفضائل، وغرض فاعلِ ذلك احتراز مودة صاحبه إلى.
        </إدخال>

        <إعادة كتابة المدخلات>
        لِمَ كان الثناء في الحضور مكروهًا حتى أَجْمَعَ الناس على زيفه؟ ولِمَ كان الثناء في الغياب محمودًا حتى تَحَبَّبَتْ إليه النفوس في كل مقام؟ أَلِأَنَّ الثناء في الحضور أقرب إلى المداهنة والخداع، وفي الغياب أقرب إلى الإخلاص والتكريم؟ أم أنَّ في الأمر معنى آخر؟ فقد قال أبو علي مسكويه، رحمه الله: إذ كان الثناء في الحضور في الغالب شهادة مُعارَة على فضائل النفس، وخديعةً للنفس بهذه الشهادة، حتى صار ذلك سببًا لانخداع المرء وتركه السعي الجاد لتحصيل الفضائل، وغرض المادح بذلك إرضاء صديقه والتودد إليه.
        </إعادة كتابة المدخلات>
    </Example_1>

    <Example_2>
        <مرجعي>
        {author_name}
        </مرجعي>

        <إدخال>
        {text}
        </إدخال>

        <إعادة كتابة المدخلات>
    """

    try:
        # Generate the rewritten text using the ALLAM model
        response = allam_model.generate_text(PROMPT_TEMPLATE)
        return response.strip()
    except Exception as e:
        logging.error(f"Error in Allam API: {e}")
        return "حدث خطأ أثناء إعادة كتابة النص. يرجى المحاولة مرة أخرى لاحقًا."


def classify_with_fine_tuned_model(text):
    """
    Classifies the author of a long text, ensuring a single prediction for the whole input.

    Args:
    - text (str): The input text.

    Returns:
    - str: The predicted author name or "Unknown".
    """
    if embedding_model is None:
        return "Unknown"
    with torch.no_grad():
        # Split the text into smaller chunks if it exceeds max length
        max_chunk_size = 2048
        chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
        
        # Generate embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            chunk_embedding = embedding_model.encode([chunk], max_length=max_chunk_size)
            embeddings.append(torch.from_numpy(chunk_embedding).to(device))
        
        # Average embeddings to create a single representation
        combined_embedding = torch.mean(torch.stack(embeddings), dim=0)

        # Perform classification on the combined embedding
        logits = classification_head_after(combined_embedding)
        predicted_idx = torch.argmax(logits, -1).item()
        return idx_to_author_mapping.get(predicted_idx, "Unknown")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("تصنيف النص", callback_data='classify')],
        [InlineKeyboardButton("إعادة كتابة النص بأسلوب آخر", callback_data='rewrite')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Define the image path or URL
    image_path = "image.jpg"  # Replace with the path to your uploaded image
    # Or use a URL if hosting the image online:
    # image_url = "https://example.com/image.png"

    # Send the image
    try:
        await context.bot.send_photo(
            chat_id=update.effective_chat.id,
            photo=open(image_path, "rb"),  # Use `image_url` if the image is hosted online
            caption=(
                "👋 **مرحباً بك!** اختر أحد الخيارات أدناه:\n\n"
                "- تصنيف النص لتحديد المؤلف المحتمل.\n"
                "- إعادة كتابة النص بأسلوب مؤلف محدد."
            ),
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    except FileNotFoundError:
        # Fallback if the image is missing
        await update.message.reply_text(
            "⚠️ تعذر تحميل الصورة. الرجاء التأكد من وجود الصورة على المسار الصحيح."
        )
    except Exception as e:
        # Handle other errors
        await update.message.reply_text(f"⚠️ حدث خطأ أثناء إرسال الصورة: {str(e)}")



async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == 'classify':
        await query.edit_message_text("📝 يرجى إرسال النص الذي ترغب في تصنيفه لتحديد المؤلف المحتمل.")
        context.user_data['action'] = 'classify'
    elif query.data == 'rewrite':
        await query.edit_message_text(
            "✍️ **إعادة كتابة النص بأسلوب مؤلف:**\n\n"
            "1️⃣ أرسل النص الأصلي الذي تريد إعادة كتابته.\n"
            "2️⃣ بعد ذلك، سأطلب منك إدخال اسم المؤلف بأسلوبه الذي تريد إعادة الكتابة."
        )
        context.user_data['action'] = 'rewrite'
        context.user_data['rewrite_step'] = 1


def split_text(text, max_length=4000):
    """
    Splits a long text into smaller chunks, each with a maximum length.

    Args:
    - text (str): The text to be split.
    - max_length (int): The maximum length of each chunk.

    Returns:
    - List[str]: A list of text chunks.
    """
    chunks = []
    while len(text) > max_length:
        # Find the last space within the limit to avoid breaking words
        split_at = text.rfind(' ', 0, max_length)
        if split_at == -1:  # No space found, split at max_length
            split_at = max_length
        chunks.append(text[:split_at])
        text = text[split_at:].strip()
    chunks.append(text)  # Add the remaining text
    return chunks

async def handle_user_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_text = update.message.text
    action = context.user_data.get('action')

    if action == 'classify':
        # Step 1: Classify the author for the entire text
        predicted_author = classify_with_fine_tuned_model(user_text)
        response_text = f"🔍 **تصنيف المؤلف**: النص يشبه إلى حد كبير أسلوب المؤلف: **{predicted_author}**."
        await update.message.reply_text(response_text, parse_mode='Markdown')

    elif action == 'rewrite':
        rewrite_step = context.user_data.get('rewrite_step', 1)

        if rewrite_step == 1:
            # Save the original text and prompt for the author's name
            context.user_data['original_text'] = user_text
            await update.message.reply_text("🖋 **الخطوة الثانية:**\nيرجى الآن إدخال اسم المؤلف الذي تريد إعادة كتابة النص بأسلوبه.")
            context.user_data['rewrite_step'] = 2

        elif rewrite_step == 2:
            # Perform the rewrite based on the original text and provided author name
            original_text = context.user_data.get('original_text')
            author_name = user_text
            rewritten_text = rewrite_text_with_allam(author_name.strip(), original_text.strip())

            # Define the image path or URL
            image_path = "image.jpg"  # Local image path
            # Or use an online image URL: image_url = "https://example.com/image.jpg"

            # Escape special characters in MarkdownV2
            response_text = (
                f"📝 **النص المُعاد كتابته بأسلوب {escape_markdown(author_name.strip(), version=2)}:**\n\n"
                f"{escape_markdown(rewritten_text, version=2)}"
            )

            try:
                # Send the image with the caption
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=open(image_path, "rb"),  # Replace with `photo=image_url` for online images
                    caption=response_text,
                    parse_mode='MarkdownV2',
                )
            except FileNotFoundError:
                # Handle missing image file
                await update.message.reply_text("⚠ عذرًا، تعذر العثور على الصورة المطلوبة.")
            except Exception as e:
                # Handle other errors
                await update.message.reply_text(f"⚠ حدث خطأ أثناء إرسال الصورة: {str(e)}")

            # Reset the rewrite step
            context.user_data['rewrite_step'] = 1

        else:
            await update.message.reply_text("⚠ يرجى اختيار إجراء قبل إرسال النص.")



# Main bot setup
def main():
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_user_text))
    application.add_handler(CallbackQueryHandler(button_handler))

    application.run_polling()


if __name__ == '__main__':
    main()
