#!/usr/bin/env python3
"""
Test script to verify the Cartesia LiveKit TTS integration works correctly
"""

import asyncio
import logging
from cartesia_livekit_tts import CartesiaTTSClient, TTSAudioOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_livekit_style_usage():
    """Test the LiveKit-style TTS usage"""
    logger.info("Testing LiveKit-style usage...")
    
    client = CartesiaTTSClient(
        api_key="test-key",  # You would use a real API key
        sample_rate=16000
    )
    
    try:
        # Test synthesize method (non-streaming)
        logger.info("Testing synthesize method...")
        text = "Hello, this is a test of the LiveKit-style TTS integration."
        stream = client.synthesize(text)
        
        audio_count = 0
        async for audio in stream:
            audio_count += 1
            logger.info(f"Received audio frame {audio_count}: {len(audio.frame.data)} bytes, final={audio.is_final}")
            if audio_count >= 3:  # Limit for testing
                break
                
        logger.info(f"Synthesize test completed with {audio_count} audio frames")
        
        # Test stream method (streaming)
        logger.info("Testing stream method...")
        stream_obj = client.stream()
        
        # Push text incrementally
        words = ["Hello", "world", "this", "is", "streaming", "TTS"]
        for word in words:
            stream_obj.push_text(word + " ")
            await asyncio.sleep(0.1)  # Simulate real-time input
            
        stream_obj.flush()
        stream_obj.end_input()
        
        stream_count = 0
        async for audio in stream_obj:
            stream_count += 1
            logger.info(f"Received streaming audio {stream_count}: {len(audio.frame.data)} bytes, final={audio.is_final}")
            if stream_count >= 3:  # Limit for testing
                break
                
        logger.info(f"Stream test completed with {stream_count} audio frames")
        
    except Exception as e:
        logger.error(f"LiveKit-style test failed: {e}")
    finally:
        await client.aclose()


async def test_legacy_compatibility():
    """Test backward compatibility with legacy methods"""
    logger.info("Testing legacy compatibility...")
    
    client = CartesiaTTSClient(
        api_key="test-key",
        sample_rate=16000
    )
    
    # Set up callback to capture TTSAudioOutput
    captured_outputs = []
    
    async def capture_output(tts_output: TTSAudioOutput):
        captured_outputs.append(tts_output)
        logger.info(f"Captured TTSAudioOutput: seq={tts_output.sequence_number}, final={tts_output.is_final}")
    
    client.add_tts_output_callback(capture_output)
    
    try:
        # Test legacy generate method
        logger.info("Testing legacy generate method...")
        await client.generate("Hello, this is a test of legacy compatibility.")
        
        # Wait a bit for generation to complete
        await asyncio.sleep(1)
        
        logger.info(f"Legacy generate test completed with {len(captured_outputs)} captured outputs")
        
        # Test legacy direct_generate method
        logger.info("Testing legacy direct_generate method...")
        captured_outputs.clear()
        
        direct_count = 0
        async for tts_output in client.direct_generate("This is a direct generation test."):
            direct_count += 1
            logger.info(f"Direct generate output {direct_count}: seq={tts_output.sequence_number}, final={tts_output.is_final}")
            if direct_count >= 3:  # Limit for testing
                break
                
        logger.info(f"Direct generate test completed with {direct_count} outputs")
        
    except Exception as e:
        logger.error(f"Legacy compatibility test failed: {e}")
    finally:
        await client.aclose()


async def test_configuration_options():
    """Test various configuration options"""
    logger.info("Testing configuration options...")
    
    # Test with different configurations
    configs = [
        {"sample_rate": 16000, "voice_id": "voice1"},
        {"sample_rate": 22050, "voice_id": "voice2"},
        {"sample_rate": 24000, "voice_id": "voice3"},
    ]
    
    for i, config in enumerate(configs):
        logger.info(f"Testing configuration {i+1}: {config}")
        
        client = CartesiaTTSClient(
            api_key="test-key",
            **config
        )
        
        try:
            # Test basic functionality
            stream = client.synthesize(f"Configuration test {i+1}")
            
            frame_count = 0
            async for audio in stream:
                frame_count += 1
                logger.info(f"Config {i+1} - Frame {frame_count}: sample_rate={audio.frame.sample_rate}")
                if frame_count >= 2:  # Limit for testing
                    break
                    
        except Exception as e:
            logger.error(f"Configuration test {i+1} failed: {e}")
        finally:
            await client.aclose()


async def main():
    """Run all tests"""
    logger.info("Starting Cartesia LiveKit TTS integration tests...")
    
    try:
        await test_livekit_style_usage()
        await asyncio.sleep(0.5)
        
        await test_legacy_compatibility()
        await asyncio.sleep(0.5)
        
        await test_configuration_options()
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())