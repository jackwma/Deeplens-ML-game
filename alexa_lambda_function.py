from __future__ import print_function
import boto3
import time
import json
import datetime  
import random
from botocore.vendored import requests
from datetime import timedelta

globalEvent = None

# --------------- Response Builders ------------------
def build_speechlet_response(title, output, reprompt_text, should_end_session):
    return {
        'outputSpeech': {
            'type': 'SSML',
            'ssml': "<speak>{}</speak>".format(output)
        },
        'card': {
            'type': 'Simple',
            'title': title,
            'content': output
        },
        'reprompt': {
            'outputSpeech': {
                'type': 'SSML',
                'ssml': "<speak>{}</speak>".format(reprompt_text)
            }
        },
        'shouldEndSession': should_end_session
    }
    
def build_response(session_attributes, speechlet_response):
    return {
        'version': '1.0',
        'sessionAttributes': session_attributes,
        'response': speechlet_response
    }
    
def sendProgressiveResponse(text):
    print("progressive response for {}".format(text))
    
    apiAccessToken = globalEvent['context']['System']['apiAccessToken']
    apiEndpoint = globalEvent['context']['System']['apiEndpoint']
    requestId = globalEvent['request']['requestId']
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer {}".format(apiAccessToken)
    }
    
    body = {
        "header": {
            "requestId": requestId
        },
        "directive": {
            "type": "VoicePlayer.Speak",
            "speech": "<speak>{}</speak>".format(text)
        }
    }
    
    url = "{}/v1/directives".format(apiEndpoint)
    progressive = requests.post(url, json=body, headers=headers)

def get_session_attributes(session):
    return session.get('attributes', {})
    
# --------------- Response Handlers ------------------
def unknown_response():
    return build_response(None, build_speechlet_response("Unknown Operation", "I don't know what to do with that", None, False))

def handle_launch_request():
    """ If we wanted to initialize the session to have some attributes we could add those here
    """
    
    session_attributes = {"state": "Welcome"}
    card_title = "Welcome"
    speech_output = "Welcome to Quick Draw game. If you're not familiar with the game, please say tell me the rules. If you already know the rules, do you want to get started now?."
    reprompt_text = "Tell me to get started or tell you the rules."
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

def handle_help_intent(session):
    """ If the user asks for the rule of the game, this is where we will explain
    """

    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "Help"
    card_title = "Help"
    speech_output = "In Quick Draw, you can have up to 4 friends to play. In each game, there are 6 rounds, and in each round, "\
    "I will ask you to draw an object, and you will have 12 seconds to draw and 8 seconds for me to guess what you're drawing. "\
    "After that 8 seconds, if what you're drawing matches what I asked, you'll get 10 points. Otherwise, you'll get nothing. "\
    "Whoever has the highest score after 6 rounds will win the game. If you want me to repeat the rules, say repeat the rules. "\
    "Otherwise, do you want to get started now?"
    
    reprompt_text = "Tell you the rules."
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))
        
def handle_yes_intent(intent, session):
    session_attributes = get_session_attributes(session)
    if session_attributes['state'] == "Welcome" or session_attributes['state'] == "Help": 
        return build_getting_started_response(intent, session)
    elif session_attributes['state'] == "WinState" or session_attributes['state'] == "LoseState" or session_attributes['state'] == "NumberOfPlayers":
        return handle_start_game_intent(intent, session)
    elif session_attributes['state'] == "StartGame" or session_attributes['state'] == "AskAgain":
        return handle_start_guessing_intent(intent, session)
    return unknown_response()
        
def handle_no_intent(intent, session):
    session_attributes = get_session_attributes(session)
    if (session_attributes['state'] == "Welcome" or session_attributes['state'] == "NumberOfPlayers" or session_attributes['state'] == "WinState" or session_attributes['state'] == "LoseState"):
        return handle_session_end_request()
    elif session_attributes['state'] == "StartGame":
        return handle_not_ready_for_guessing_intent(intent, session)
    elif session_attributes['state'] == "AskAgain":
        return build_never_ready_response(session, session_attributes["playersScore"])
    return unknown_response()
        
def handle_session_end_request():
    card_title = "Session Ended"
    speech_output = "Ok, I hope you had fun. Have a nice day! "
    should_end_session = True
    return build_response({}, build_speechlet_response(
        card_title, speech_output, None, should_end_session))

def handle_fallback(intent, session):
    card_title = "Huh?"
    speech_output = "I'm not sure if I understand. "
    session_attributes = get_session_attributes(session)
    if (session_attributes["state"] == "Welcome"):
        speech_output += "Say tell me the rules if you're not familiar with the game. Otherwise, do you want to get started now?"
    elif (session_attributes["state"] == "Help"):
        speech_output += "Say repeat the rules if you want to hear the rules again. Otherwise, do you want to get started now?"
    elif (session_attributes["state"] == "GettingStarted"):
        speech_output += "How many players are there playing the game?"
    elif (session_attributes["state"] == "NumberOfPlayers"):
        speech_output += "Are you all ready to play?"
    elif (session_attributes["state"] == "StartGame" or session_attributes["state"] == "AskAgain"):
        speech_output += "Can I start guessing now?"
    elif (session_attributes["state"] == "WinState" or session_attributes["state"] == "LoseState"):
        currentRound = session_attributes["round"]
        currentTurn = session_attributes["turn"]
        numberPlayers = session_attributes["numberPlayers"]
        if currentTurn != numberPlayers:
            speech_output += "Player {}. Are you ready for your turn?".format(currentTurn + 1)
        else:
            speech_output += "Round {} has finished. Are you all ready for the next round?".format(currentRound)
            
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, None, should_end_session))
        
def build_getting_started_response(intent, session):
    #After the player says yes for being ready to get started
    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "GettingStarted"
    card_title = "NumberOfPlayers"
    speech_output = "How many players are there playing the game?"
    reprompt_text = "How many players?"
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))
 
def handle_number_of_players_intent(intent, session):
    """ 
    Ask player(s) if they're ready to play
    """
    numberPlayers = int(intent["slots"]["numberPlayers"]["value"])
    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "NumberOfPlayers"
    session_attributes["numberPlayers"] = numberPlayers
    session_attributes["usedObjects"] = []
    
    playersScore = [0] * numberPlayers
    session_attributes["playersScore"] = playersScore
    
    card_title = "Number of Players"

    speech_output = "I see there are {} players. Are you all ready to play?".format(numberPlayers)
     
    reprompt_text = "Are you ready to play?"
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))
        
def handle_start_game_intent(intent, session):
    """ 
    User is ready to start the game. Tell them their object to draw, wait 4 seconds, and then
    ask if Alexa should start guessing.
    """
    listOfObjects = ['Eiffel Tower', 'airplane', 'alarm clock', 'ambulance', 'angel', 'ant', 'apple', 
    'baseball bat', 'bicycle', 'birthday cake', 'bowtie', 'bucket', 'camera', 'candle', 'car', 'carrot', 
    'donut', 'elephant', 'eye', 'fish', 'flower', 'fork', 'hat', 'hexagon', 'hourglass', 'ice cream', 
    'ladder', 'lollipop', 'mug', 'paper clip', 'skateboard', 'snowflake', 'sun', 'sword', 'television', 'umbrella']
    objectDrawing = None
    session_attributes = get_session_attributes(session)
    
    while objectDrawing == None:
        objectToDraw =  listOfObjects[random.randrange(len(listOfObjects))]
        if (objectToDraw not in session_attributes["usedObjects"]):
            objectDrawing = objectToDraw
    
    session_attributes["usedObjects"].append(objectDrawing)
    session_attributes["state"] = "StartGame"
    session_attributes["objectDrawing"] = objectDrawing
    
    #Setting up the player's turn and the round number
    if ("round" in session_attributes.keys() and "turn" in session_attributes.keys()):
        if (session_attributes["turn"] == session_attributes["numberPlayers"]):
            session_attributes["round"] += 1
            session_attributes["turn"] = 1
        else:
            session_attributes["turn"] += 1
    else:
        session_attributes["round"] = 1
        session_attributes["turn"] = 1
    
    currentRound = session_attributes["round"]
    currentTurn = session_attributes["turn"]
    card_title = "StartGame"
    speech_output = "This is round {} and player {}'s turn. Your object to draw is <break time=\"0.5s\"/> {}. Start drawing now! <break time=\"4s\"/> Can I start guessing now?".format(currentRound, currentTurn, objectDrawing)
    reprompt_text = "Can I start guessing now?"
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))      
        
def handle_not_ready_for_guessing_intent(intent, session):
    """ 
    Ask if user is ready to start guessing again
    """
    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "AskAgain"
    card_title = "Can I Start"
    speech_output = "<break time=\"4s\"/> Can I start guessing now?"
    reprompt_text = "Can I start guessing now?"
    should_end_session = False
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

def handle_start_guessing_intent(intent, session):
    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "StartGuessing"
    reprompt_text = None
    speech_output=""
    should_end_session=False
    
    kinesis = boto3.client('kinesis', region_name='us-east-1')
    shard_id = 'shardId-000000000000'
    shard_it = None
    sequenceNumber=""
    
    objectDrawing = session_attributes["objectDrawing"] 
    
    if len(sequenceNumber)==0:   
        print("NO SEQUENCE NUMBER, GETTING OBJECTS DETECTED IN LAST FEW MINUTES")
        shard_it = kinesis.get_shard_iterator(
            StreamName='AggregatedDataOutput', 
            ShardId=shard_id, 
            ShardIteratorType='LATEST')['ShardIterator']
        print('shard_it = ' + shard_it)
            
    else:
        print("GOT SEQUENCE NUMBER, GETTING OBJECTS DETECTED - " + sequenceNumber)
        shard_it = kinesis.get_shard_iterator(
            StreamName='AggregatedDataOutput', 
            ShardId=shard_id, 
            ShardIteratorType='AFTER_SEQUENCE_NUMBER',
            StartingSequenceNumber=sequenceNumber)['ShardIterator']
             
    foundObject = False
    startTime = time.time()
    while time.time() - startTime < 8:
        out = kinesis.get_records(ShardIterator=shard_it, Limit=1)
        shard_it = out["NextShardIterator"]
        print('out = ', out)
        
        if (len(out["Records"])) > 0:
            print('records = ', out["Records"])
            theObject = json.loads(out["Records"][0]["Data"])
            print('kinesis object = ', theObject)
            objectName = max(theObject, key=theObject.get)
            print('Best Prob. object = ', objectName)
            sendProgressiveResponse("I see a " + objectName)
            if objectName == objectDrawing:
                foundObject = True
                break
            
        time.sleep(0.2)
    
    numberPlayers = session_attributes["numberPlayers"]
    currentTurn = session_attributes["turn"]
    playersScore = session_attributes["playersScore"]

    if foundObject:
        playersScore[currentTurn - 1] += 10
        session_attributes["playersScore"] = playersScore
        return build_win_response(session, playersScore)
    else:
        return build_lose_response(session, playersScore)
        
def build_end_of_player_turn_output(currentTurn, currentRound, numberPlayers, playersScore, audio, salutation):
    speech_output = ""
    should_end_session = False
    
    intro = "{} {} Your current score is {}. <break time=\"0.8s\"/>".format(audio, salutation, playersScore[currentTurn - 1])
    
    if currentTurn != numberPlayers:
        speech_output = "{} Player {}. Are you ready for your turn?".format(intro, currentTurn + 1)    
    else:
        # end of round
        if (currentRound != 2):
            # end of intermediate round
            speech_output = "{} Round {} has finished. ".format(intro, currentRound)
            for i in range(1, numberPlayers + 1):
                speech_output += "Player {}'s score is {}. <break time=\"0.4s\"/> ".format(i, playersScore[i - 1])
            speech_output += "Are you all ready for the next round?"
        else:
            # end of final round
            speech_output = "{} Round {} has finished, and that's the last round. ".format(intro, currentRound)
            for i in range(1, numberPlayers + 1):
                speech_output += "Player {}'s total score is {}. <break time=\"0.4s\"/> ".format(i, playersScore[i - 1])
            maxScore = max(playersScore)
            winningPlayers = [i + 1 for i, j in enumerate(playersScore) if j == maxScore]
            if (len(winningPlayers) > 1):
                # handling a tie game
                winners = ""
                for player in winningPlayers:
                    winners += "Player {}. ".format(player)
                
                if maxScore > 0:    
                    speech_output += "So there is a tie. The winners are {} <audio src=\"soundbank://soundlibrary/human/amzn_sfx_crowd_applause_01\"/>"\
                    "Congratulations! You won the game! Thank you for playing Quick Draw. I hope you had an amazing time.".format(winners)
                else:
                    speech_output += "So nobody won the game. <audio src=\"soundbank://soundlibrary/alarms/beeps_and_bloops/boing_03\"/> Thank you for playing Quick Draw. "\
                    "I hope you had an amazing time."
            else:
                speech_output += "So the winner is: player {}. <audio src=\"soundbank://soundlibrary/human/amzn_sfx_crowd_applause_01\"/> "\
                "Congratulations! You won the game! Thank you for playing Quick Draw. I hope you had an amazing time.".format(winningPlayers[0])
            should_end_session = True
            
    return speech_output, should_end_session
            
def build_win_response(session, playersScore):
    """ 
    Say that the player wins
    """

    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "WinState"
    session_attributes["playersScore"] = playersScore
    
    numberPlayers = session_attributes["numberPlayers"]
    currentTurn =  session_attributes["turn"]
    currentRound = session_attributes["round"]

    card_title = "Winner"
    reprompt_text = "Winning."
    
    speech_output, should_end_session = build_end_of_player_turn_output(currentTurn, currentRound, numberPlayers, playersScore,
        "<audio src=\"soundbank://soundlibrary/human/amzn_sfx_crowd_applause_01\"/>", "Nice job! You won this round.")
            
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))
 
def build_lose_response(session, playersScore):
    """ 
    Say that the player loses
    """
    
    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "LoseState"
    card_title = "Loser"
    reprompt_text = "Losing."
    should_end_session = False
    
    numberPlayers = session_attributes["numberPlayers"]
    currentTurn =  session_attributes["turn"]
    currentRound = session_attributes["round"]
    
    session_attributes["playersScore"] = playersScore
    
    speech_output, should_end_session = build_end_of_player_turn_output(currentTurn, currentRound, numberPlayers, playersScore,
        "<audio src=\"soundbank://soundlibrary/alarms/buzzers/buzzers_09\"/>", "Oopsie. I didn't see what I asked you to draw.")
        
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))

def build_never_ready_response(session, playersScore):
    """ 
    Say that the player loses for the player is not ready for alexa to guess
    """

    session_attributes = get_session_attributes(session)
    session_attributes["state"] = "LoseState"
    card_title = "Loser"
        
    reprompt_text = "Losing"
    should_end_session = False
    
    numberPlayers = session_attributes["numberPlayers"]
    currentTurn =  session_attributes["turn"]
    currentRound = session_attributes["round"]
    
    session_attributes["playersScore"] = playersScore
    
    speech_output, should_end_session = build_end_of_player_turn_output(currentTurn, currentRound, numberPlayers, playersScore,
        "<audio src=\"soundbank://soundlibrary/alarms/buzzers/buzzers_09\"/>", "You're taking too long. Your turn is over.")
    
    return build_response(session_attributes, build_speechlet_response(
        card_title, speech_output, reprompt_text, should_end_session))
        
# --------------- Events ------------------

def on_session_started(session_started_request, session):
    """ Called when the session starts """

    print("on_session_started requestId=" + session_started_request['requestId']
          + ", sessionId=" + session['sessionId'])

def on_launch(launch_request, session):
    """ Called when the user launches the skill without specifying what they
    want
    """

    print("on_launch requestId=" + launch_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # Dispatch to your skill's launch
    return handle_launch_request()

def on_intent(intent_request, session):
    """ Called when the user specifies an intent for this skill """

    print("on_intent requestId=" + intent_request['requestId'] +
          ", sessionId=" + session['sessionId'])

    intent = intent_request['intent']
    intent_name = intent_request['intent']['name']
    
    # Dispatch to your skill's intent handlers
    if intent_name == "StartGame":
        return handle_start_game_intent(intent, session)
    elif intent_name == "StartGuessing":
        return handle_start_guessing_intent(intent, session)
    elif intent_name == "NumberOfPlayers":  
        return handle_number_of_players_intent(intent, session)
    elif intent_name == "AMAZON.HelpIntent":
        return handle_help_intent(session)
    elif intent_name == "AMAZON.YesIntent":
        return handle_yes_intent(intent, session)
    elif intent_name == "AMAZON.NoIntent":
        return handle_no_intent(intent, session)
    elif intent_name == "AMAZON.CancelIntent" or intent_name == "AMAZON.StopIntent":
        return handle_session_end_request()
    elif intent_name == "AMAZON.FallbackIntent":
        return handle_fallback(intent, session)
    else:
        raise ValueError("Invalid intent")

def on_session_ended(session_ended_request, session):
    """ Called when the user ends the session.

    Is not called when the skill returns should_end_session=true
    """
    print("on_session_ended requestId=" + session_ended_request['requestId'] +
          ", sessionId=" + session['sessionId'])
    # add cleanup logic here

# --------------- Main handler ------------------

def lambda_handler(event, context):
    """ Route the incoming request based on type (LaunchRequest, IntentRequest,
    etc.) The JSON body of the request is provided in the event parameter.
    """
    global globalEvent
    globalEvent = event
    
    print("event.session.application.applicationId=" +
          event['session']['application']['applicationId'])

    """
    Uncomment this if statement and populate with your skill's application ID to
    prevent someone else from configuring a skill that sends requests to this
    function.
    """
    # if (event['session']['application']['applicationId'] !=
    #         "amzn1.echo-sdk-ams.app.[unique-value-here]"):
    #     raise ValueError("Invalid Application ID")

    if event['session']['new']:
        on_session_started({'requestId': event['request']['requestId']},
                           event['session'])

    if event['request']['type'] == "LaunchRequest":
        return on_launch(event['request'], event['session'])
    elif event['request']['type'] == "IntentRequest":
        return on_intent(event['request'], event['session'])
    elif event['request']['type'] == "SessionEndedRequest":
        return on_session_ended(event['request'], event['session'])

