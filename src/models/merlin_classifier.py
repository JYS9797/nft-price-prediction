\
import torch
import torch.nn as nn
from transformers import BertModel, ViTModel


class MERLINClassifier(nn.Module):
    """
    Multimodal classifier: BERT text + ViT image with attention fusion,
    augmented with collection-level numeric features (floor/avg prices).

    Output: logits over `num_classes`.
    """
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        image_model_name: str = "google/vit-base-patch16-224-in21k",
        num_classes: int = 3,
    ):
        super().__init__()
        self.text_model = BertModel.from_pretrained(text_model_name)
        self.text_fc = nn.Linear(768, 256)

        self.image_model = ViTModel.from_pretrained(image_model_name)
        self.image_fc = nn.Linear(768, 256)

        self.collection_fc = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
        )

        self.attention_fc = nn.Linear(256, 1)

        self.classifier = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids, attention_mask, pixel_values, floor_price, avg_price):
        # text
        text_emb = self.text_model(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        text_emb = self.text_fc(text_emb)

        # image
        image_emb = self.image_model(pixel_values=pixel_values).pooler_output
        image_emb = self.image_fc(image_emb)

        # attention fusion
        combined_emb = torch.stack([text_emb, image_emb], dim=1)
        attn_weights = torch.softmax(self.attention_fc(combined_emb), dim=1)
        fused_emb = (combined_emb * attn_weights).sum(dim=1)

        # collection numeric features
        collection_features = torch.cat([floor_price, avg_price], dim=1)
        collection_emb = self.collection_fc(collection_features)

        # final classifier
        final_emb = torch.cat([fused_emb, collection_emb], dim=1)
        logits = self.classifier(final_emb)
        return logits
