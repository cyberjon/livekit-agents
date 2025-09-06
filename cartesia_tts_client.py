import asyncio
import base64
import json
import logging
import weakref
from dataclasses import dataclass, replace
from typing import AsyncGenerator, Union, Any, Optional

from cartesia import AsyncCartesia
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from openai import AsyncStream
from pydantic import BaseModel
import aiohttp

# LiveKit imports
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions, 
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit import rtc

# Your existing imports - adjust these paths as needed for your codebase
import app.settings.settings as aps
from app.api.models.ws_data import TTSAudioOutput
from app.utils.types import XobotTypes
from app.tts.tts_forwarder import AudioWordChunk, TTSForwarder
from app.tts.utils import text_chunker, TTSInterruptedError, get_current_timestamp
from app.event.config.events_config import EventType as ConfigEventType

backend_settings = aps.get_settings()
LOGGER = logging.getLogger(__name__)


class CartesiaConfig(BaseModel):
    """Configuration for Cartesia TTS client"""
    api_key: str
    voice_id: str = "5c42302c-194b-4d0c-ba1a-8cb485c84ab9"
    model_id: str = "sonic"
    sample_rate: int = 22050
    chunk_size: int = 32000
    buffer_size: int = 32768
    max_queue_size: int = 50


@dataclass
class _TTSOptions:
    model: str
    encoding: str
    sample_rate: int
    voice: str
    api_key: str
    language: str
    base_url: str

    def get_http_url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def get_ws_url(self, path: str) -> str:
        return f"{self.base_url.replace('http', 'ws', 1)}{path}"


class CartesiaTTSClient(tts.TTS):
    """Cartesia TTS Client using LiveKit approach while emitting TTSAudioOutput"""

    def __init__(self, 
                 api_key: Optional[str] = None,
                 voice_id: str = "5c42302c-194b-4d0c-ba1a-8cb485c84ab9",
                 model_id: str = "sonic",
                 sample_rate: int = 16000,
                 http_session: Optional[aiohttp.ClientSession] = None):
        
        # Initialize LiveKit TTS base class
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=False,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )
        
        # Initialize Cartesia client
        self.config = CartesiaConfig(
            api_key=api_key or backend_settings.cartesia_api_key,
            voice_id=voice_id,
            model_id=model_id,
            sample_rate=sample_rate
        )
        self.client = AsyncCartesia(api_key=self.config.api_key)
        
        # LiveKit-style options
        self._opts = _TTSOptions(
            model=model_id,
            encoding="pcm_s16le",
            sample_rate=sample_rate,
            voice=voice_id,
            api_key=self.config.api_key,
            language="en",
            base_url="https://api.cartesia.ai"
        )
        
        # Session management
        self._session = http_session
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=300,
            mark_refreshed_on_get=True,
        )
        self._streams = weakref.WeakSet()
        
        # Event emission setup
        self._tts_output_callbacks: list = []
        self._language_code = 'en'
        self._played_text = None
        self.output_format = None
        self.forwarder: Optional = None
        
        # Task management
        self._generation_task: Optional[asyncio.Task] = None
        self.stop_requested = asyncio.Event()

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        url = self._opts.get_ws_url(f"/tts/websocket?api_key={self._opts.api_key}&cartesia_version=2024-06-10")
        return await asyncio.wait_for(session.ws_connect(url), timeout)

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse) -> None:
        await ws.close()

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> 'ChunkedStreamWrapper':
        """LiveKit-style synthesize method that returns a ChunkedStream"""
        return ChunkedStreamWrapper(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> 'SynthesizeStreamWrapper':
        """LiveKit-style stream method that returns a SynthesizeStream"""
        return SynthesizeStreamWrapper(tts=self, conn_options=conn_options)

    # Legacy methods for backward compatibility
    async def generate(
        self,
        text: Union[str, StreamingAgentChatResponse, AsyncStream],
        **kwargs
    ):
        """Legacy generate method - starts TTS generation task"""
        if self._generation_task and not self._generation_task.done():
            LOGGER.warning("Generation task already running. Cancelling previous task.")
            await self.stop_generation()

        self._generation_task = asyncio.create_task(self._do_generate(text, **kwargs))

    async def _do_generate(
        self,
        text: Union[str, StreamingAgentChatResponse, AsyncStream],
        language_code: str = "en-US",
        telephone_provider: XobotTypes = XobotTypes.WEB,
        voice_id: str = None,
        correlation_id: str = None
    ) -> None:
        """Legacy generation implementation using LiveKit synthesize"""
        start_time_overall = get_current_timestamp()
        interrupted = False
        final_source_texts = []

        self.stop_requested.clear()
        self._language_code = language_code.split("-")[0]
        self._configure_output_format(telephone_provider)
        self._played_text = ""

        # Emit TTS_STARTED event
        initial_text_segment = "Streaming input"
        if isinstance(text, str):
            initial_text_segment = text[:100] + ("..." if len(text) > 100 else "")
        await self._emit_tts_started(text=initial_text_segment, provider="Cartesia", correlation_id=correlation_id)

        try:
            # Use LiveKit synthesize method
            stream = self.synthesize(str(text) if isinstance(text, str) else "streaming input")
            
            sequence_number = 0
            async for synthesized_audio in stream:
                if self.stop_requested.is_set():
                    raise TTSInterruptedError("TTS interrupted")
                
                # Convert LiveKit SynthesizedAudio to TTSAudioOutput
                tts_output = TTSAudioOutput(
                    current_audio_chunk=synthesized_audio.frame.data,
                    final_audio_chunks=[synthesized_audio.frame.data],
                    start_timestamp=get_current_timestamp(),
                    end_timestamp=0,
                    sequence_number=sequence_number,
                    source_texts=[synthesized_audio.delta_text] if synthesized_audio.delta_text else [],
                    is_final=synthesized_audio.is_final
                )
                
                # Emit TTSAudioOutput for backward compatibility
                await self._emit_tts_chunk(tts_output, correlation_id=correlation_id)
                sequence_number += 1

        except TTSInterruptedError:
            LOGGER.info("Cartesia TTS generation interrupted.")
            interrupted = True
        except asyncio.CancelledError:
            LOGGER.info("Cartesia TTS generation task was cancelled.")
            interrupted = True
        except Exception as e:
            LOGGER.error(f"Error during Cartesia TTS generation: {str(e)}", exc_info=True)
            await self._emit_tts_error(e, correlation_id=correlation_id)
            interrupted = True
        finally:
            if self._played_text and self._played_text.strip():
                final_source_texts = self._played_text.strip().split()
            duration_ms = get_current_timestamp() - start_time_overall
            await self._emit_tts_completed(
                duration_ms=duration_ms,
                interrupted=interrupted,
                source_texts=final_source_texts,
                correlation_id=correlation_id
            )
            self._generation_task = None
            self.forwarder = None

    async def direct_generate(
        self,
        text: Union[str, StreamingAgentChatResponse, AsyncStream],
        voice_id: str = None,
        telephone_provider: XobotTypes = XobotTypes.WEB,
        language_code: str = "en-US",
    ) -> AsyncGenerator[TTSAudioOutput, Any]:
        """Legacy direct_generate method using LiveKit stream"""
        self.stop_requested.clear()
        self._language_code = language_code.split("-")[0]
        self._configure_output_format(telephone_provider)

        sample_rate = self.output_format.get("sample_rate", 16000)
        self.forwarder = TTSForwarder(sample_rate=sample_rate, telephone_provider=telephone_provider)

        try:
            # Use LiveKit stream method
            stream_obj = self.stream()
            
            # Push text to stream
            if isinstance(text, str):
                stream_obj.push_text(text)
            else:
                async for chunk in text_chunker(text):
                    if self.forwarder.stop_requested.is_set():
                        raise TTSInterruptedError("TTS interrupted by forwarder signal")
                    stream_obj.push_text(chunk)
            
            stream_obj.flush()
            stream_obj.end_input()
            
            sequence_number = 0
            async for synthesized_audio in stream_obj:
                if self.forwarder.stop_requested.is_set():
                    raise TTSInterruptedError("TTS interrupted")
                    
                tts_output = TTSAudioOutput(
                    current_audio_chunk=synthesized_audio.frame.data,
                    final_audio_chunks=[synthesized_audio.frame.data],
                    start_timestamp=get_current_timestamp(),
                    end_timestamp=0,
                    sequence_number=sequence_number,
                    source_texts=[synthesized_audio.delta_text] if synthesized_audio.delta_text else [],
                    is_final=synthesized_audio.is_final
                )
                yield tts_output
                sequence_number += 1
                
        except TTSInterruptedError:
            LOGGER.info("Cartesia TTS direct_generate interrupted.")
        except Exception as e:
            LOGGER.error(f"Error during Cartesia TTS direct_generate: {str(e)}", exc_info=True)
        finally:
            self.forwarder = None

    def _configure_output_format(self, telephone_provider: XobotTypes):
        """Configure output format based on telephone provider"""
        if telephone_provider in (XobotTypes.WEB, XobotTypes.KNOWLARIY):
            self.output_format = {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
            }
        else:  # Default to ulaw_8000 for Twilio
            self.output_format = {
                "container": "raw",
                "encoding": "pcm_mulaw",
                "sample_rate": 8000,
            }

    def _calculate_chunk_duration(self, chunk: bytes, sample_rate: int, channels: int = 1, sample_width: int = 2) -> float:
        """Calculate duration of an audio chunk in seconds"""
        if not chunk:
            return 0.0
        # For PCM s16le, sample width is 2 bytes. For mulaw, it's 1 byte.
        if self.output_format and self.output_format['encoding'] == 'pcm_mulaw':
            sample_width = 1
        num_samples = len(chunk) / (channels * sample_width)
        duration = num_samples / sample_rate
        return duration

    async def stop_generation(self):
        """Stop the ongoing generation task"""
        self.stop_requested.set()
        if self.forwarder:
            await self.forwarder.stop_generation()
        if self._generation_task and not self._generation_task.done():
            self._generation_task.cancel()
            try:
                await self._generation_task
            except asyncio.CancelledError:
                pass

    # Event emission methods for backward compatibility
    async def _emit_tts_chunk(self, tts_output: TTSAudioOutput, correlation_id: str = None):
        """Emit TTS chunk for backward compatibility"""
        for callback in self._tts_output_callbacks:
            try:
                await callback(tts_output)
            except Exception as e:
                LOGGER.error(f"Error in TTS output callback: {e}")

    async def _emit_tts_started(self, text: str, provider: str, correlation_id: str = None):
        """Emit TTS started event"""
        # Implement your event emission logic here
        pass

    async def _emit_tts_completed(self, duration_ms: int, interrupted: bool, source_texts: list, correlation_id: str = None):
        """Emit TTS completed event"""
        # Implement your event emission logic here
        pass

    async def _emit_tts_error(self, error: Exception, correlation_id: str = None):
        """Emit TTS error event"""
        # Implement your event emission logic here
        pass

    def add_tts_output_callback(self, callback):
        """Add callback for TTSAudioOutput emission"""
        self._tts_output_callbacks.append(callback)

    async def aclose(self) -> None:
        """Close the TTS client"""
        await self.stop_generation()
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStreamWrapper(tts.ChunkedStream):
    """Wrapper around LiveKit ChunkedStream that also emits TTSAudioOutput"""
    
    def __init__(self, *, tts: CartesiaTTSClient, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: CartesiaTTSClient = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the TTS synthesis using Cartesia HTTP API"""
        json_data = {
            "model_id": self._opts.model,
            "transcript": self.input_text,
            "voice": {
                "mode": "id",
                "id": self._opts.voice
            },
            "output_format": {
                "container": "raw",
                "encoding": self._opts.encoding,
                "sample_rate": self._opts.sample_rate,
            },
            "language": self._opts.language,
        }

        try:
            async with self._tts._ensure_session().post(
                self._opts.get_http_url("/tts/bytes"),
                headers={
                    "X-API-Key": self._opts.api_key,
                    "Cartesia-Version": "2024-06-10",
                },
                json=json_data,
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                resp.raise_for_status()

                output_emitter.initialize(
                    request_id=utils.shortuuid(),
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type="audio/pcm",
                )

                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)

                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e


class SynthesizeStreamWrapper(tts.SynthesizeStream):
    """Wrapper around LiveKit SynthesizeStream that also emits TTSAudioOutput"""
    
    def __init__(self, *, tts: CartesiaTTSClient, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: CartesiaTTSClient = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        """Run the streaming TTS synthesis using Cartesia WebSocket API"""
        request_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type="audio/pcm",
            stream=True,
        )

        async def _input_task() -> None:
            context_id = utils.shortuuid()
            base_pkt = {
                "model_id": self._opts.model,
                "voice": {
                    "mode": "id",
                    "id": self._opts.voice
                },
                "output_format": {
                    "container": "raw",
                    "encoding": self._opts.encoding,
                    "sample_rate": self._opts.sample_rate,
                },
                "language": self._opts.language,
            }
            
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    continue
                    
                token_pkt = base_pkt.copy()
                token_pkt["context_id"] = context_id
                token_pkt["transcript"] = data + " "
                token_pkt["continue"] = True
                self._mark_started()
                await ws.send_str(json.dumps(token_pkt))

            # Send end packet
            end_pkt = base_pkt.copy()
            end_pkt["context_id"] = context_id
            end_pkt["transcript"] = " "
            end_pkt["continue"] = False
            await ws.send_str(json.dumps(end_pkt))

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            current_segment_id: str | None = None
            while True:
                msg = await ws.receive()
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise APIStatusError(
                        "Cartesia connection closed unexpectedly", request_id=request_id
                    )

                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                segment_id = data.get("context_id")
                if current_segment_id is None:
                    current_segment_id = segment_id
                    output_emitter.start_segment(segment_id=segment_id)
                    
                if data.get("data"):
                    b64data = base64.b64decode(data["data"])
                    output_emitter.push(b64data)
                elif data.get("done"):
                    output_emitter.end_input()
                    break
                elif data.get("type") == "error":
                    raise APIError(f"Cartesia returned error: {data}")

        try:
            async with self._tts._pool.connection(timeout=self._conn_options.timeout) as ws:
                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_recv_task(ws)),
                ]

                try:
                    await asyncio.gather(*tasks)
                finally:
                    await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            raise APIConnectionError() from e