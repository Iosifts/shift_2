# Imports

def create_prompt(prompt_dict, audience_difficulty, raw=True):
    """
    Create a prompt for the image description task.
    """
    # general prompt
    prompt = f'You are a professional museum curator with the task of creating a concise and engaging description for an image. \
        If the informations are available, the description should always start with title, author, date.'
    # audience rules: combinations of categories and their rules, according to the audience difficulty
    prompt += create_audience_prompt(prompt_dict, audience_difficulty)
    # context and question prompting
    prompt += f"Try to answer the following question at the end of the description: {prompt_dict['question']}" if prompt_dict['question'].strip() else ""
    prompt += f"Following additional information is given: {prompt_dict['context']}" if prompt_dict['context'].strip() else ""
    # few-shot prompting
    prompt += f"An example of an image description is given. Please follow the general structure of \
        the example to create the image description: {prompt_dict['example']}" if prompt_dict['example'].strip() else ""
    # hyperparameter prompting
    prompt += create_hyperparameter_prompt(prompt_dict)

    # post-process the prompt
    if not raw:
        # Specific format
        prompt = f"""user\n\n<image>\n{prompt}\n\nassistant\n\n"""

    print(prompt)

    return prompt

def create_hyperparameter_prompt(prompt_dict):
    """
    Create further optional hyperparameters rules based on the prompt dictionary.
    """
    rules = f'Please incorporate the following language rules in your description: '
    rules += f"""
    Use the following sentiment/tone: {prompt_dict['sentiment']},
    Use the following language formality: {prompt_dict['formality']},
    Finally, create the description in the following language: {prompt_dict['language']}. 
    """
    return rules

def create_audience_prompt(prompt_dict, audience_difficulty):
    """
    Create audience rules based on the prompt dictionary.
    """
    audience_rules = f"The description should be tailored to the following audience-specific categories: \n"
    if prompt_dict['audience_age'][0] is not None:
        audience_rules += f"Age Group: {prompt_dict['audience_age'][0]}. "
    if prompt_dict['audience_abilities'] is not None:
        audience_rules += f"Restricted/Enhanced Abilities: {', '.join(prompt_dict['audience_abilities'])}. "
    #audience_rules += f"Education: {prompt_dict['audience_education'][0]}. "
    if prompt_dict['audience_art'][0] is not None:
        audience_rules += f"Art Affiliation: {prompt_dict['audience_art'][0]}. "
    audience_rules += f"Following are the rules for the specific audience categories: \n"

    # ---------------------------------------------------------------------------------------------
    # Category 1: Age
    if "Childs"in prompt_dict['audience_age']:
        audience_age_short = ("Ensure the description is engaging for children.")
        audience_age_medium = ("When addressing children, it is important to use language that is fun, engaging, and easy to understand, while keeping the instructions simple and interactive.")
        audience_age_long = ("When addressing children, it is important to use language that is fun, engaging, and easy to understand, while keeping the instructions simple and interactive. \
                             Audience Rules for Children: Use simple, child-friendly language and avoid complicated words. Keep sentences short, direct, and easy to follow. Use a playful, \
                             encouraging tone to keep children engaged. Incorporate fun visuals, emojis, or characters to make instructions lively. Break down tasks into small, simple steps with clear examples. \
                             Encourage participation through interactive elements like games or quizzes. Use repetition and reinforcement to ensure understanding. \
                             Offer praise and positive feedback to motivate continued interaction. Avoid overwhelming them with too much information at once; focus on one thing at a time. \
                             Include safety measures or reminders when necessary (e.g., 'Make sure an adult is with you!').")
    elif "Elderly" in prompt_dict['audience_age']:
        audience_age_short = ("Ensure the description is easy to follow for elderly people.")
        audience_age_medium = ("For elderly users, the language should be clear, respectful, and supportive, considering potential cognitive or sensory challenges.")
        audience_age_long = ("For elderly users, the language should be clear, respectful, and supportive, considering potential cognitive or sensory challenges. \
                             Audience Rules for Elderly: Use polite and respectful language with a friendly tone. Avoid jargon or complex terminology; use simple, \
                             familiar words. Increase font size or use bold to emphasize key points. Speak slowly, and allow for longer pauses or reflection in any \
                             interactive prompts. Provide examples or context to clarify difficult concepts. Use step-by-step instructions and repeat important information. \
                             Make navigation intuitive and buttons easy to click or tap. Be mindful of potential visual impairments and avoid overly bright or harsh colors.")
    elif "Adults" in prompt_dict['audience_age']:
        audience_age_short = ("Ensure the description is suitable for adults.")
        audience_age_medium = ("Ensure the description is suitable for adults.")
        audience_age_long = ("Ensure the description is suitable for adults.")
    else:
        audience_age_short = ("Ensure the description is easy to follow for all age groups.")
        audience_age_medium = ("Ensure the description is easy to follow for all age groups.")
        audience_age_long = ("Ensure the description is easy to follow for all age groups.")

    # ---------------------------------------------------------------------------------------------
    # Category 2: Restricted/Enhanced Abilities (non-mutually exclusive category).

    if prompt_dict['audience_abilities'] is not None:
        if "Visually Impaired" in prompt_dict['audience_abilities']:
            audience_abilities_short = ("Ensure the description follows alt text rules for digital accessibility. Include any text present in the image. Describe visible elements first without interpretations, and summarize possible interpretations at the end.")
            audience_abilities_medium = ("Ensure the description follows alt text rules for digital accessibility. Include any text present in the image. Describe visible elements first without interpretations, and summarize possible interpretations at the end.")
            audience_abilities_long = ("For visually impaired individuals, focus on providing accessible instructions that can easily be understood through screen readers and other assistive technology. Ensure a clear structure and alternative descriptions. \
                                    Audience Rules for Visually Impaired People: \
                                    - Ensure all text content is compatible with screen readers. \
                                    - Provide alt text for all images, describing key details succinctly and clearly. \
                                    - Use headers and structured layouts so content is easily navigable. \
                                    - Avoid relying solely on visual elements to convey important information (use text-based explanations). \
                                    - Use descriptive, precise language to replace visual cues (e.g., 'Click the large button labeled ‘Submit’ at the bottom of the page'). \
                                    - Keep instructions concise and avoid overly complex language. - Ensure all interactive elements are accessible via keyboard (not just a mouse or touchscreen). \
                                    - Provide audio descriptions for any multimedia content. - Minimize distractions like pop-ups or animations that could interfere with screen reader navigation. \
                                    - Use clear and explicit labels for all buttons, links, and form fields.")
        if "ADHD" in prompt_dict['audience_abilities']:
            audience_abilities_short += ("Ensure the description is easy to read for people with ADHD, using clear and direct sentences.")
            audience_abilities_medium += ("People with ADHD may benefit from clear, concise, and engaging instructions. Try to avoid overwhelming them with too much detail at once.")
            audience_abilities_long += ("People with ADHD may benefit from clear, concise, and engaging instructions. Try to avoid overwhelming them with too much detail at once. Audience Rules for ADHD: \
                                    - Use short, clear, and direct sentences. \
                                    - Break instructions into smaller, manageable chunks or steps. \
                                    - Avoid long paragraphs; use bullet points when possible. \
                                    - Include visual or interactive elements to keep engagement high. \
                                    - Use simple language and avoid over-complicating ideas. \
                                    - Provide frequent cues or reminders to stay on track. \
                                    - Limit distractions (unnecessary details or additional links). \
                                    - Offer praise or positive reinforcement to maintain motivation.")
        if "Dyslexia" in prompt_dict['audience_abilities']:
            audience_abilities_short += ("Ensure the description is easy to read for people with dyslexia by using clear formatting and fonts.")
            audience_abilities_medium += ("Ensure the description is easy to read for people with dyslexia by using clear formatting, short sentences, and familiar language.")
            audience_abilities_long += ("People with dyslexia benefit from clear, concise, and accessible instructions. Avoid overwhelming them with too much detail or complex language. Audience Rules for Dyslexia: \
                                    - Use large, readable fonts and ample spacing between lines. \
                                    - Break down the content into small, digestible pieces. \
                                    - Use bullet points and numbered lists to structure information. \
                                    - Avoid italicized text or excessive use of capital letters, which can be harder to read. \
                                    - Use clear, simple language, and avoid technical jargon. \
                                    - Highlight key information with bolding or underlining. \
                                    - Provide optional audio instructions for those who prefer to listen. \
                                    - Use visual aids or illustrations when possible to reinforce the text.")
        else:
            audience_abilities_short = ("For normally-abled people, the instructions should be clear, concise, and engaging.")
            audience_abilities_medium = ("For normally-abled people, the instructions should be clear, concise, and engaging.")
            audience_abilities_long = ("For normally-abled people, the instructions should be clear, concise, and engaging.")
    else:
        audience_abilities_short = ("For normally-abled people, the instructions should be clear, concise, and engaging.")
        audience_abilities_medium = ("For normally-abled people, the instructions should be clear, concise, and engaging.")
        audience_abilities_long = ("For normally-abled people, the instructions should be clear, concise, and engaging.")

    # ---------------------------------------------------------------------------------------------
    # Category 3: Education
    # if "High Education" in prompt_dict['audience_education']:
    #     audience_education_short = ("Ensure the description is precise, technical, and intellectually engaging.")
    #     audience_education_medium = ("For highly educated individuals, focus on providing clear, insightful information that addresses the product's advanced features, benefits, and broader implications.")
    #     audience_education_long = ("For a highly educated audience, the instructions should emphasize complexity and depth. Audience Rules for Highly Educated Individuals: \
    #                                Begin with an introduction that highlights the sophisticated and innovative aspects of the service. \
    #                                Assume a high level of prior knowledge and comfort with technical or abstract concepts. \
    #                                Use precise, analytical language and reference relevant research or theoretical frameworks where appropriate. \
    #                                Provide in-depth explanations of features and benefits, and encourage critical thinking. \
    #                                Highlight how the product aligns with their intellectual or professional goals, offering detailed examples or case studies. \
    #                                Include links to additional resources or scholarly material to satisfy their curiosity. \
    #                                Provide a clear call to action with options for further exploration, like an advanced trial or demo.")
    # elif "Medium Education" in prompt_dict['audience_education']:
    #     audience_education_short = ("Ensure the description is clear, informative, and straightforward.")
    #     audience_education_medium = ("For individuals with a moderate level of education, focus on providing clear and balanced instructions that are neither too simplistic nor overly technical.")
    #     audience_education_long = ("For an audience with a medium level of education, the instructions should be informative and accessible. Audience Rules for Medium-Educated Individuals: \
    #                                Start with a clear and concise introduction that outlines the practical benefits of the product. \
    #                                Assume a reasonable level of prior knowledge, but avoid highly technical language. \
    #                                Use clear and practical language to explain each step, ensuring understanding without oversimplifying. \
    #                                Highlight the value of the product in helping them achieve practical or everyday goals. \
    #                                Offer helpful examples or scenarios to illustrate the usefulness of the product. \
    #                                Include references to further resources for those interested in learning more. \
    #                                Encourage them to take action with a direct, motivating call to action and, if applicable, offer a demo or trial to try out the service.")
    # elif "Low Education" in prompt_dict['audience_education']:
    #     audience_education_short = ("Ensure the description is simple, direct, and easy to understand.")
    #     audience_education_medium = ("For individuals with lower levels of formal education, focus on providing clear and simple instructions with practical language that anyone can understand.")
    #     audience_education_long = ("For an audience with a lower level of education, the instructions should prioritize simplicity and clarity. Audience Rules for Low-Education Individuals: \
    #                                Begin with a simple, straightforward introduction that clearly explains the main benefits of the service. \
    #                                Assume minimal prior knowledge and avoid technical or complex terminology. \
    #                                Use plain, conversational language to explain each step clearly and directly. \
    #                                Break down instructions into small, easy-to-follow steps, avoiding unnecessary details or complexity. \
    #                                Highlight how the product can solve practical problems or improve their everyday lives. \
    #                                Offer visual aids, simple tutorials, or step-by-step guides to assist with understanding. \
    #                                Provide reassurances and a clear, encouraging call to action that feels accessible and easy to follow. \
    #                                If applicable, offer an easy-to-use demo or free trial to help them get started without commitment.")
    # else:
    #     audience_education_short = ("Ensure the instructions are clear, concise, and suitable for a general audience.")
    #     audience_education_medium = ("For a general audience, the instructions should be clear, concise, and easy to follow, with enough information to ensure understanding across different education levels.")
    #     audience_education_long = ("For a general audience, the instructions should be accessible, clear, and engaging. Audience Rules for General Audiences: \
    #                                Begin with a simple introduction that explains the purpose and benefits of the product in a clear way. \
    #                                Avoid complex jargon and technical terms, focusing on making the instructions easy to understand. \
    #                                Use friendly, approachable language that balances professionalism and accessibility. \
    #                                Break down the steps into clear, manageable instructions that are easy to follow. \
    #                                Highlight how the product addresses common needs or improves everyday tasks. \
    #                                Offer examples or illustrations to help visualize the product’s functionality. \
    #                                Include a clear call to action that invites them to try the product, and consider offering a free trial or demo for exploration.")

    # ---------------------------------------------------------------------------------------------
    # Category 4: Art Affiliation

    if prompt_dict['audience_art'] is not None:

        if "Art-Professional" in prompt_dict['audience_art']:
            audience_art_short = ("Ensure the description is concise, clear, and uses technical terms correctly.")
            audience_art_medium = ("For professionals familiar with the product or service, focus on clear, precise language that highlights the unique features and benefits without over-explaining basic concepts.")
            audience_art_long = ("For professionals familiar with the product or service, the instructions should be concise and to the point. Audience Rules for Professionals: \
                                Start with a brief overview of the product's advanced features and industry-specific benefits. \
                                Assume prior knowledge and use technical or industry-specific terms where necessary. \
                                Focus on efficiency and expertise in the language to appeal to their experience. \
                                Offer detailed insights into the product’s unique selling points and advanced functionalities. \
                                Provide a direct call to action with an emphasis on how this product can enhance their professional workflow. \
                                Highlight any professional certifications, integrations, or customization options. \
                                Consider offering an advanced demo or trial for deeper exploration.")
        elif "Art-Hobbyist" in prompt_dict['audience_art']:
            audience_art_short = ("Keep the description engaging and accessible, with an enthusiastic tone.")
            audience_art_medium = ("For hobbyists, who may have some familiarity with the product or service, focus on excitement and inspiration while providing clear, step-by-step guidance.")
            audience_art_long = ("For hobbyists, the instructions should focus on engagement and inspiration, while providing clear guidance. Audience Rules for Hobbyists: \
                                Start with an enthusiastic introduction that connects the product to their interests. \
                                Assume some familiarity, but avoid overly technical language; offer tips to enhance their skills. \
                                Use an encouraging and motivational tone that inspires creativity and experimentation. \
                                Offer clear, step-by-step instructions with suggestions for customization or creative use. \
                                Emphasize how the product can elevate their hobby, making it more enjoyable or efficient. \
                                Highlight community resources, tutorials, or forums to help them learn more. \
                                Encourage action with a call to explore more features, and consider offering a trial version or starter kit.")
        elif "Art-Amateur" in prompt_dict['audience_art']:
            audience_art_short = ("Ensure the description is friendly, simple, and encouraging for beginners.")
            audience_art_medium = ("For amateurs, who may be new to the product or service, focus on building confidence and explaining things in simple, non-technical terms.")
            audience_art_long = ("For amateurs, the instructions should focus on simplicity and encouragement. Audience Rules for Amateurs: \
                                Start with a friendly introduction that emphasizes ease of use and accessibility. \
                                Assume no prior knowledge and avoid technical terms, explaining everything step-by-step. \
                                Use a supportive and patient tone to build their confidence. \
                                Provide detailed explanations with background information where needed, and avoid overwhelming the reader. \
                                Highlight how the product can help them start or improve their experience, making it approachable. \
                                Include visual aids, tutorials, or beginner-friendly resources to help them get started. \
                                Offer reassurances, FAQs, and a clear call to action that encourages them to try the product. \
                                Provide an easy-to-access demo or free trial to lower the barrier to entry.")
        else:
            audience_art_short = ("Make sure the instructions are clear, easy to follow, and engaging for a general audience.")
            audience_art_medium = ("For a general audience, the instructions should be simple, concise, and engaging, with enough detail to ensure understanding without overwhelming the reader.")
            audience_art_long = ("For a general audience, the instructions should be clear, concise, and engaging. Audience Rules for a General Audience: \
                                Start with a brief introduction that explains the benefits and purpose of the product. \
                                Avoid jargon and keep the language conversational and approachable. \
                                Ensure each step is clearly explained, without assuming too much prior knowledge. \
                                Use a friendly tone that balances professionalism with approachability. \
                                Highlight ease of use and how the product addresses common needs or problems. \
                                Provide visual aids or examples where necessary to improve understanding. \
                                Offer a direct call to action, encouraging the audience to try or learn more about the product. \
                                If applicable, provide links to demos, free trials, or further resources.")
    else:
        audience_art_short = ("Make sure the instructions are clear, easy to follow, and engaging for a general audience.")
        audience_art_medium = ("For a general audience, the instructions should be simple, concise, and engaging, with enough detail to ensure understanding without overwhelming the reader.")
        audience_art_long = ("For a general audience, the instructions should be clear, concise, and engaging. Audience Rules for a General Audience: \
                             Start with a brief introduction that explains the benefits and purpose of the product. \
                             Avoid jargon and keep the language conversational and approachable. \
                             Ensure each step is clearly explained, without assuming too much prior knowledge. \
                             Use a friendly tone that balances professionalism with approachability. \
                             Highlight ease of use and how the product addresses common needs or problems. \
                             Provide visual aids or examples where necessary to improve understanding. \
                             Offer a direct call to action, encouraging the audience to try or learn more about the product. \
                             If applicable, provide links to demos, free trials, or further resources.")

    # ---------------------------------------------------------------------------------------------

    if audience_difficulty == "Low":
        audience_rules += audience_age_short
        audience_rules += audience_abilities_short
        #audience_rules += audience_education_short
        audience_rules += audience_art_short

    elif audience_difficulty == "Medium":
        audience_rules += audience_age_medium
        audience_rules += audience_abilities_medium
        #audience_rules += audience_education_medium
        audience_rules += audience_art_medium

    elif audience_difficulty == "High":
        audience_rules += audience_age_long
        audience_rules += audience_abilities_long
        #audience_rules += audience_education_long
        audience_rules += audience_art_long


    return audience_rules