# CUPID: ControlNet-based Unique Pair Image Designer
**Generate Matching Couple Avatars with AI**  
CUPID is an AI system that automatically creates visually consistent couple avatars from a single input image. Our pipeline combines:
- **ControlNet-guided Stable Diffusion** for pose-aligned generation  
- **GPT-4o prompting** for detailed textual guidance  
- **AdaIN style transfer** for enhanced consistency  

✔️ Trained on 636 high-quality avatar pairs  
✔️ Preserves structural correspondence through image flipping  
✔️ Style-consistent outputs via adaptive normalization  


**Use Cases**: Social media avatars, game characters, personalized gifts  

**Running**: add your LLM API key to .env, then run demo.py or 启动指南.md if you want to deploy

**Training**: run train.py

# Citation

    @misc{zhang2023adding,
      title={Adding Conditional Control to Text-to-Image Diffusion Models}, 
      author={Lvmin Zhang and Anyi Rao and Maneesh Agrawala},
      booktitle={IEEE International Conference on Computer Vision (ICCV)}
      year={2023},
    }
