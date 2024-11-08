<template>
  <div class="browse">
    <h1>This is the video browse page</h1>

    <div class="flexcontainer">
      <img class="timet" src="/home/miked/Pictures/Screenshots/timetable.png"/>
      <img class="timet" src="/home/miked/Pictures/Screenshots/Screenshot from 2024-10-10 17-36-38.png"/>
      <img class="timet" src="/home/miked/Pictures/Screenshots/Screenshot from 2024-10-11 10-46-17.png"/>
    </div>
  </div>
</template>

<style>
@media (min-width: 1024px) {
  .browse {
    margin: 0 2rem;
    /* min-height: 100vh;
    display: flex;
    align-items: center; */
  }
  .flexcontainer {
    display: flex
  }
  .timet {
    width: 30rem;
  }
}
</style>
